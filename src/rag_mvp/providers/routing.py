"""Role-specific provider routing, compatibility, readiness, and fallback."""

from __future__ import annotations

from dataclasses import dataclass

from rag_mvp.providers.errors import (
    ProviderConfigurationError,
    ProviderError,
    ProviderOperationError,
)
from rag_mvp.providers.models import (
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingSpaceIdentity,
    GenerationRequest,
    GenerationResult,
    ModelAttempt,
    ProviderCallContext,
    ProviderErrorCategory,
    ProviderRole,
    RerankRequest,
    RerankResult,
    RoleReadiness,
    RoutedRerankResult,
    RoutedResult,
    RouteMetadata,
)
from rag_mvp.providers.protocols import (
    AttemptRecorder,
    EmbeddingProvider,
    GenerationProvider,
    NullAttemptRecorder,
    RerankingProvider,
)
from rag_mvp.providers.resilience import RetryPolicy, execute_with_resilience


@dataclass(frozen=True, slots=True)
class ProviderRoute[P]:
    route_id: str
    provider: P
    retry_policy: RetryPolicy

    def __post_init__(self) -> None:
        if not self.route_id.strip():
            raise ValueError("route_id must not be empty")


EmbeddingRoute = ProviderRoute[EmbeddingProvider]
GenerationRoute = ProviderRoute[GenerationProvider]
RerankingRoute = ProviderRoute[RerankingProvider]


class ModelProviderRouter:
    """Ordered provider routes with role-specific fallback semantics."""

    def __init__(
        self,
        *,
        embedding_routes: tuple[EmbeddingRoute, ...] = (),
        generation_routes: tuple[GenerationRoute, ...] = (),
        reranking_routes: tuple[RerankingRoute, ...] = (),
        recorder: AttemptRecorder | None = None,
    ) -> None:
        self._embedding_routes = tuple(embedding_routes)
        self._generation_routes = tuple(generation_routes)
        self._reranking_routes = tuple(reranking_routes)
        self._recorder = recorder or NullAttemptRecorder()
        self._validate_routes()

    def _validate_routes(self) -> None:
        route_ids: set[str] = set()
        route_groups: tuple[tuple[object, ...], ...] = (
            self._embedding_routes,
            self._generation_routes,
            self._reranking_routes,
        )
        for route_group in route_groups:
            for raw_route in route_group:
                route = raw_route
                if not isinstance(route, ProviderRoute):
                    raise ProviderConfigurationError("invalid_provider_route")
                if route.route_id in route_ids:
                    raise ProviderConfigurationError("duplicate_provider_route")
                route_ids.add(route.route_id)
        for route in self._embedding_routes:
            if not isinstance(route.provider, EmbeddingProvider):
                raise ProviderConfigurationError("embedding_capability_missing")
        for route in self._generation_routes:
            if not isinstance(route.provider, GenerationProvider):
                raise ProviderConfigurationError("generation_capability_missing")
        for route in self._reranking_routes:
            if not isinstance(route.provider, RerankingProvider):
                raise ProviderConfigurationError("reranking_capability_missing")

    @property
    def readiness(self) -> tuple[RoleReadiness, ...]:
        return (
            RoleReadiness(
                ProviderRole.EMBEDDING,
                bool(self._embedding_routes),
                None if self._embedding_routes else "embedding_provider_unavailable",
            ),
            RoleReadiness(
                ProviderRole.GENERATION,
                bool(self._generation_routes),
                None if self._generation_routes else "generation_provider_unavailable",
            ),
            RoleReadiness(
                ProviderRole.RERANKING,
                bool(self._reranking_routes),
                None if self._reranking_routes else "reranking_provider_unavailable",
            ),
        )

    @property
    def qa_ready(self) -> bool:
        readiness = {status.role: status.ready for status in self.readiness}
        return readiness[ProviderRole.EMBEDDING] and readiness[ProviderRole.GENERATION]

    async def embed(
        self,
        request: EmbeddingRequest,
        context: ProviderCallContext,
        *,
        required_space: EmbeddingSpaceIdentity,
    ) -> RoutedResult[EmbeddingResult]:
        compatible = [
            (position, route)
            for position, route in enumerate(self._embedding_routes)
            if route.provider.identity == required_space
        ]
        if not compatible:
            raise ProviderOperationError(
                ProviderErrorCategory.INCOMPATIBLE_RESPONSE,
                fallback_eligible=False,
            )

        attempts: list[ModelAttempt] = []
        last_error: ProviderOperationError | None = None
        for position, route in compatible:
            is_fallback = position > 0
            metadata = RouteMetadata(
                route.route_id,
                ProviderRole.EMBEDDING,
                required_space.model_identity,
            )

            async def operation(
                selected: EmbeddingRoute = route,
            ) -> EmbeddingResult:
                result = await selected.provider.embed(request, context)
                if result.identity != required_space or len(result.vectors) != len(request.texts):
                    raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
                return result

            try:
                attempted = await execute_with_resilience(
                    operation,
                    context=context,
                    route=metadata,
                    policy=route.retry_policy,
                    is_fallback=is_fallback,
                    recorder=self._recorder,
                )
            except ProviderOperationError as error:
                attempts.extend(error.attempts)
                last_error = error
                if not error.fallback_eligible:
                    break
                continue
            attempts.extend(attempted.attempts)
            return RoutedResult(attempted.value, tuple(attempts), is_fallback)

        if last_error is None:
            raise ProviderOperationError(ProviderErrorCategory.UNAVAILABLE, tuple(attempts))
        raise ProviderOperationError(
            last_error.category,
            tuple(attempts),
            retryable=last_error.retryable,
            fallback_eligible=last_error.fallback_eligible,
        )

    async def generate(
        self,
        request: GenerationRequest,
        context: ProviderCallContext,
    ) -> RoutedResult[GenerationResult]:
        if not self._generation_routes:
            raise ProviderOperationError(ProviderErrorCategory.UNAVAILABLE)
        attempts: list[ModelAttempt] = []
        last_error: ProviderOperationError | None = None
        for position, route in enumerate(self._generation_routes):
            is_fallback = position > 0
            metadata = RouteMetadata(
                route.route_id,
                ProviderRole.GENERATION,
                route.provider.identity,
            )

            async def operation(
                selected: GenerationRoute = route,
            ) -> GenerationResult:
                result = await selected.provider.generate(request, context)
                if result.identity != selected.provider.identity:
                    raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
                return result

            try:
                attempted = await execute_with_resilience(
                    operation,
                    context=context,
                    route=metadata,
                    policy=route.retry_policy,
                    is_fallback=is_fallback,
                    recorder=self._recorder,
                )
            except ProviderOperationError as error:
                attempts.extend(error.attempts)
                last_error = error
                if not error.fallback_eligible:
                    break
                continue
            attempts.extend(attempted.attempts)
            return RoutedResult(attempted.value, tuple(attempts), is_fallback)

        if last_error is None:
            raise ProviderOperationError(ProviderErrorCategory.UNAVAILABLE, tuple(attempts))
        raise ProviderOperationError(
            last_error.category,
            tuple(attempts),
            retryable=last_error.retryable,
            fallback_eligible=last_error.fallback_eligible,
        )

    async def rerank(
        self,
        request: RerankRequest,
        context: ProviderCallContext,
    ) -> RoutedRerankResult:
        base_order = request.candidate_ids
        if not self._reranking_routes:
            return RoutedRerankResult(
                ordered_ids=base_order,
                attempts=(),
                applied=False,
                degraded=True,
                degradation_reason=ProviderErrorCategory.UNAVAILABLE,
            )

        attempts: list[ModelAttempt] = []
        last_category = ProviderErrorCategory.UNAVAILABLE
        for position, route in enumerate(self._reranking_routes):
            is_fallback = position > 0
            metadata = RouteMetadata(
                route.route_id,
                ProviderRole.RERANKING,
                route.provider.identity,
            )

            async def operation(
                selected: RerankingRoute = route,
            ) -> RerankResult:
                result = await selected.provider.rerank(request, context)
                if (
                    result.identity != selected.provider.identity
                    or result.prompt_version != request.prompt_version
                    or not _is_exact_permutation(result.ordered_ids, base_order)
                ):
                    raise ProviderError(ProviderErrorCategory.INCOMPATIBLE_RESPONSE)
                return result

            try:
                attempted = await execute_with_resilience(
                    operation,
                    context=context,
                    route=metadata,
                    policy=route.retry_policy,
                    is_fallback=is_fallback,
                    recorder=self._recorder,
                )
            except ProviderOperationError as error:
                attempts.extend(error.attempts)
                last_category = error.category
                if not error.fallback_eligible:
                    break
                continue
            attempts.extend(attempted.attempts)
            return RoutedRerankResult(
                ordered_ids=attempted.value.ordered_ids,
                attempts=tuple(attempts),
                applied=True,
                degraded=False,
            )

        return RoutedRerankResult(
            ordered_ids=base_order,
            attempts=tuple(attempts),
            applied=False,
            degraded=True,
            degradation_reason=last_category,
        )


def _is_exact_permutation(order: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    return (
        len(order) == len(expected)
        and len(set(order)) == len(order)
        and set(order) == set(expected)
    )
