"""Generate a fictional Chinese life-insurance policy PDF for RAG testing."""

from __future__ import annotations

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "sample_life_insurance_policy_zh.pdf"
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

NAVY = (0.05, 0.18, 0.34)
BLUE = (0.10, 0.42, 0.68)
PALE = (0.92, 0.96, 0.99)
GOLD = (0.80, 0.58, 0.18)
INK = (0.12, 0.15, 0.18)
MUTED = (0.38, 0.43, 0.48)
RED = (0.76, 0.08, 0.08)
WHITE = (1, 1, 1)
PAGE = fitz.paper_rect("a4")
MARGIN = 46


def add_fonts(page: fitz.Page) -> None:
    page.insert_font(fontname="zh", fontfile=str(FONT))
    page.insert_font(fontname="zhb", fontfile=str(FONT_BOLD))


def textbox(
    page: fitz.Page,
    rect: fitz.Rect,
    text: str,
    *,
    size: float = 10,
    color: tuple[float, float, float] = INK,
    bold: bool = False,
    align: int = fitz.TEXT_ALIGN_LEFT,
    lineheight: float = 1.35,
) -> None:
    result = page.insert_textbox(
        rect,
        text,
        fontname="zhb" if bold else "zh",
        fontsize=size,
        color=color,
        align=align,
        lineheight=lineheight,
    )
    if result < 0:
        raise RuntimeError(f"Text overflow: {text[:30]!r} ({result})")


def header(page: fitz.Page, section: str, page_no: int) -> None:
    add_fonts(page)
    page.draw_rect(fitz.Rect(0, 0, PAGE.width, 54), color=NAVY, fill=NAVY)
    textbox(page, fitz.Rect(MARGIN, 13, 300, 42), "安宁人寿 · 演示保单", size=14, color=WHITE, bold=True)
    textbox(page, fitz.Rect(300, 16, PAGE.width - MARGIN, 40), section, size=9, color=WHITE, align=fitz.TEXT_ALIGN_RIGHT)
    textbox(page, fitz.Rect(MARGIN, PAGE.height - 29, PAGE.width - MARGIN, PAGE.height - 12), f"示例文件｜非有效保险合同                              第 {page_no} 页 / 共 5 页", size=8, color=MUTED, align=fitz.TEXT_ALIGN_CENTER)


def title(page: fitz.Page, y: float, number: str, name: str) -> float:
    page.draw_rect(fitz.Rect(MARGIN, y, PAGE.width - MARGIN, y + 31), color=BLUE, fill=PALE, width=0.8)
    textbox(page, fitz.Rect(MARGIN + 12, y + 6, PAGE.width - MARGIN - 8, y + 27), f"{number}  {name}", size=12, color=NAVY, bold=True)
    return y + 42


def table(page: fitz.Page, y: float, rows: list[tuple[str, str]], *, row_h: float = 31) -> float:
    left, right = MARGIN, PAGE.width - MARGIN
    split = left + 135
    for idx, (label, value) in enumerate(rows):
        top = y + idx * row_h
        fill = PALE if idx % 2 == 0 else WHITE
        page.draw_rect(fitz.Rect(left, top, right, top + row_h), color=(0.78, 0.83, 0.87), fill=fill, width=0.5)
        page.draw_line(fitz.Point(split, top), fitz.Point(split, top + row_h), color=(0.78, 0.83, 0.87), width=0.5)
        textbox(page, fitz.Rect(left + 9, top + 7, split - 6, top + row_h - 4), label, size=9, color=NAVY, bold=True)
        textbox(page, fitz.Rect(split + 9, top + 7, right - 7, top + row_h - 4), value, size=9)
    return y + len(rows) * row_h


def bullets(page: fitz.Page, y: float, items: list[str], *, gap: float = 45) -> float:
    for item in items:
        page.draw_circle(fitz.Point(MARGIN + 5, y + 9), 2.2, color=GOLD, fill=GOLD)
        textbox(page, fitz.Rect(MARGIN + 16, y, PAGE.width - MARGIN, y + gap - 3), item, size=9.3, lineheight=1.45)
        y += gap
    return y


def cover(doc: fitz.Document) -> None:
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    add_fonts(page)
    page.draw_rect(page.rect, color=WHITE, fill=WHITE)
    page.draw_rect(fitz.Rect(0, 0, PAGE.width, 235), color=NAVY, fill=NAVY)
    page.draw_rect(fitz.Rect(0, 235, PAGE.width, 244), color=GOLD, fill=GOLD)
    textbox(page, fitz.Rect(MARGIN, 55, PAGE.width - MARGIN, 92), "安宁人寿保险股份有限公司", size=15, color=WHITE, bold=True)
    textbox(page, fitz.Rect(MARGIN, 111, PAGE.width - MARGIN, 160), "恒爱一生定期寿险", size=28, color=WHITE, bold=True)
    textbox(page, fitz.Rect(MARGIN, 164, PAGE.width - MARGIN, 198), "保险单（演示样本）", size=18, color=(0.86, 0.91, 0.96), bold=True)

    page.draw_rect(fitz.Rect(MARGIN, 286, PAGE.width - MARGIN, 421), color=(0.78, 0.83, 0.87), fill=PALE, width=0.8)
    textbox(page, fitz.Rect(MARGIN + 22, 307, PAGE.width - MARGIN - 22, 338), "保单编号：DEMO-LIFE-2026-000888", size=13, color=NAVY, bold=True)
    textbox(page, fitz.Rect(MARGIN + 22, 349, PAGE.width - MARGIN - 22, 379), "保险期间：2026年8月8日零时至2046年8月7日二十四时", size=10)
    textbox(page, fitz.Rect(MARGIN + 22, 386, PAGE.width - MARGIN - 22, 413), "基本保险金额：人民币 1,000,000.00 元", size=10, bold=True)

    page.draw_rect(fitz.Rect(MARGIN, 467, PAGE.width - MARGIN, 560), color=RED, fill=(1.0, 0.95, 0.95), width=1.1)
    textbox(page, fitz.Rect(MARGIN + 18, 486, PAGE.width - MARGIN - 18, 518), "重要声明", size=14, color=RED, bold=True, align=fitz.TEXT_ALIGN_CENTER)
    textbox(page, fitz.Rect(MARGIN + 22, 522, PAGE.width - MARGIN - 22, 552), "本文件仅用于软件演示、检索和测试，不代表任何真实承保关系，不具有法律效力。", size=10, color=RED, bold=True, align=fitz.TEXT_ALIGN_CENTER)

    textbox(page, fitz.Rect(MARGIN, 620, PAGE.width - MARGIN, 660), "签发日期：2026年8月7日", size=10, align=fitz.TEXT_ALIGN_CENTER)
    textbox(page, fitz.Rect(MARGIN, 680, PAGE.width - MARGIN, 735), "客服热线：400-000-0000（虚构）\n公司地址：中国上海市示范路88号（虚构）", size=9, color=MUTED, align=fitz.TEXT_ALIGN_CENTER)
    textbox(page, fitz.Rect(MARGIN, PAGE.height - 29, PAGE.width - MARGIN, PAGE.height - 12), "示例文件｜非有效保险合同                              第 1 页 / 共 5 页", size=8, color=MUTED, align=fitz.TEXT_ALIGN_CENTER)


def parties(doc: fitz.Document) -> None:
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    header(page, "保单信息与合同主体", 2)
    y = title(page, 79, "01", "保单基本信息")
    y = table(page, y, [
        ("保单编号", "DEMO-LIFE-2026-000888"),
        ("险种名称", "恒爱一生定期寿险（演示产品）"),
        ("合同生效日", "2026年8月8日"),
        ("保险期间", "20年"),
        ("缴费方式", "年缴，共缴10年"),
        ("首期保险费", "人民币 6,880.00 元"),
    ]) + 24
    y = title(page, y, "02", "合同主体")
    table(page, y, [
        ("投保人", "李明｜男｜1988年5月18日出生"),
        ("投保人证件", "居民身份证 310***********1234（脱敏虚构）"),
        ("被保险人", "李明｜男｜职业：软件工程师"),
        ("通讯地址", "上海市浦东新区示范大道100号（虚构）"),
        ("身故受益人", "王芳（配偶），受益比例100%"),
        ("受益顺序", "第一顺序"),
    ])


def coverage(doc: fitz.Document) -> None:
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    header(page, "保障内容", 3)
    y = title(page, 79, "03", "保险责任与给付")
    y = bullets(page, y, [
        "身故保险金：被保险人在保险期间内身故，且不属于本合同约定的责任免除情形，本公司按基本保险金额人民币1,000,000.00元给付身故保险金，本合同终止。",
        "全残保险金：被保险人在保险期间内达到本合同约定的全残标准，本公司按基本保险金额人民币1,000,000.00元给付全残保险金，本合同终止。",
        "身故保险金与全残保险金仅给付其中一项，二者不重复给付。",
    ], gap=69) + 14
    y = title(page, y, "04", "等待期与犹豫期")
    y = bullets(page, y, [
        "等待期为合同生效之日起90日。因意外伤害导致的保险事故不受等待期限制。等待期内因疾病身故或全残，按约定退还已交保险费，本合同终止。",
        "犹豫期为投保人签收本合同之日起15日。犹豫期内申请解除合同，本公司在扣除不超过人民币10元的工本费后退还已收保险费。",
    ], gap=73) + 12
    y = title(page, y, "05", "保费与宽限期")
    bullets(page, y, [
        "续期保险费应在每年8月8日前缴纳。未按时缴费的，自应缴日起享有60日宽限期；宽限期内发生保险事故，本公司承担保险责任，但会从给付金额中扣除欠缴保险费。",
    ], gap=78)


def exclusions(doc: fitz.Document) -> None:
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    header(page, "责任免除与理赔", 4)
    y = title(page, 79, "06", "责任免除")
    y = bullets(page, y, [
        "投保人对被保险人的故意杀害或故意伤害；",
        "被保险人故意犯罪，或者抗拒依法采取的刑事强制措施；",
        "被保险人自本合同成立或者效力恢复之日起2年内自杀，但被保险人自杀时为无民事行为能力人的除外；",
        "被保险人主动吸食或者注射毒品、酒后驾驶、无合法有效驾驶证驾驶，或驾驶无合法有效行驶证的机动车；",
        "战争、军事冲突、暴乱、武装叛乱，或核爆炸、核辐射及核污染。",
    ], gap=46) + 8
    y = title(page, y, "07", "保险金申请")
    bullets(page, y, [
        "申请人应提交保险合同、申请人有效身份证明、被保险人的死亡证明或符合约定的全残鉴定材料，以及本公司合理要求的其他证明。",
        "材料齐全后，本公司应及时作出核定；情形复杂的，原则上在30日内作出核定。属于保险责任的，在达成给付协议后按约定支付保险金。",
        "保险事故发生后，申请人应及时通知本公司。故意或因重大过失未及时通知导致事故性质、原因或损失程度难以确定的，本公司对无法确定的部分不承担给付责任。",
    ], gap=62)


def terms(doc: fitz.Document) -> None:
    page = doc.new_page(width=PAGE.width, height=PAGE.height)
    header(page, "合同服务与签署", 5)
    y = title(page, 79, "08", "合同变更、解除与争议处理")
    y = bullets(page, y, [
        "联系方式、受益人等信息发生变化时，投保人应及时书面通知本公司办理变更。受益人变更依法需要被保险人同意的，应取得其同意。",
        "犹豫期后投保人可申请解除合同。本公司收到解除申请及所需材料后，退还合同当时的现金价值；解除可能造成经济损失。",
        "因履行本合同发生争议，双方可协商解决；协商不成的，可依法向有管辖权的人民法院提起诉讼。",
        "本演示保单未列事项以正式保险条款及法律法规为准；但本文件本身不构成真实保险合同或承保承诺。",
    ], gap=58) + 14
    y = title(page, y, "09", "签署确认（演示）")
    page.draw_rect(fitz.Rect(MARGIN, y, PAGE.width - MARGIN, y + 150), color=(0.78, 0.83, 0.87), fill=(0.98, 0.99, 1), width=0.7)
    textbox(page, fitz.Rect(MARGIN + 18, y + 17, PAGE.width - MARGIN - 18, y + 56), "投保人确认已阅读并理解保险责任、责任免除、犹豫期、退保损失等重要内容。", size=9.5)
    textbox(page, fitz.Rect(MARGIN + 18, y + 76, 275, y + 108), "投保人签名：________________", size=9.5)
    textbox(page, fitz.Rect(310, y + 76, PAGE.width - MARGIN - 18, y + 108), "被保险人签名：________________", size=9.5)
    textbox(page, fitz.Rect(MARGIN + 18, y + 116, PAGE.width - MARGIN - 18, y + 143), "签署日期：2026年8月7日", size=9.5)
    textbox(page, fitz.Rect(MARGIN, y + 180, PAGE.width - MARGIN, y + 230), "安宁人寿保险股份有限公司（虚构）\n电子签章：仅供演示", size=10, color=NAVY, bold=True, align=fitz.TEXT_ALIGN_CENTER)


def main() -> None:
    if not FONT.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("Required Chinese fonts were not found")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    cover(doc)
    parties(doc)
    coverage(doc)
    exclusions(doc)
    terms(doc)
    metadata = {
        "title": "恒爱一生定期寿险保险单（演示样本）",
        "author": "安宁人寿保险股份有限公司（虚构）",
        "subject": "用于RAG系统测试的虚构中文人寿保险保单",
        "keywords": "演示, 人寿保险, 保单, RAG测试, 非有效合同",
        "creator": "RAG Demo sample document generator",
    }
    doc.set_metadata(metadata)
    doc.save(OUTPUT, garbage=4, deflate=True)
    doc.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
