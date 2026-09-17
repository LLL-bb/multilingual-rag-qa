"""生成内置示例语料。

语料刻意做成"中英各四篇、主题互不重叠":

    中文: 校园规章 / 图书馆借阅规则 / 宿舍管理条例 / 奖学金评定办法
    英文: 课程手册 / 餐饮菜单 / 图书馆开放政策 / IT 使用政策

每篇都写到几百至上千字,保证能切出多个块 —— 语料如果一篇只切出一块,
检索评测就退化成"选文档",失去区分度。

同时覆盖四种解析格式,用来验证 loaders.py 的每条分支:

    .md   校园规章、图书馆借阅规则、course_handbook、library_policy
    .txt  宿舍管理条例、dining_menu
    .docx 奖学金评定办法
    .pdf  it_policy

PDF 用内置 Helvetica 手工构造,所以必须是纯 ASCII —— 这也限制了
it_policy 只能写成英文。中文 PDF 需要嵌入 CID 字体,不值得为示例语料引这个依赖。

用法:
    python scripts/build_demo_corpus.py [--out data/raw] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap  # noqa: E402

PROJECT_ROOT = bootstrap()

CAMPUS_RULES = """# 校园行为规范

## 第一章　总则

第一条　为维护校园秩序,保障师生人身与财产安全,特制定本规范。

第二条　凡在本校校园内活动的人员,包括学生、教职工、访客及施工单位人员,均适用本规范。

## 第二章　公共秩序

第三条　校园内所有室内区域禁止吸烟,包括教学楼、图书馆、宿舍楼、食堂及体育场馆。

第四条　禁止在校园内饮酒、赌博、打架斗殴,禁止携带管制刀具及其他危险物品进入校园。

第五条　校内禁止携带宠物,导盲犬及其他经学校登记的工作犬除外。

第六条　未经许可不得在校园内张贴海报、悬挂横幅、散发传单或进行商业推广活动。

第七条　不得在校园道路、消防通道、无障碍坡道上停放自行车、电动车或堆放杂物。

## 第三章　教学与科研场所

第八条　进入教学楼、图书馆、实验室须出示有效学生证或教工证,配合安保人员查验。

第九条　实验室操作须严格遵守安全规程,使用危险化学品须两人以上在场并登记。

第十条　考试期间不得携带与考试无关的物品进入考场,不得有抄袭、替考等行为。

## 第四章　网络与信息

第十一条　校园网络仅限用于教学、科研及办公,不得用于非法活动或大规模下载。

第十二条　不得擅自接入或改动校园网络设备,不得扫描、攻击校内信息系统。

## 第五章　处分

第十三条　违反本规范的,视情节轻重给予口头警告、通报批评、纪律处分,
或移交司法机关处理。
"""

LIBRARY_BORROWING = """# 图书馆借阅规则

## 一、借阅册数与期限

本科生一次最多可借 10 册,借期 30 天。

研究生一次最多可借 20 册,借期 60 天。

教师及研究人员一次最多可借 30 册,借期 90 天。

工具书、期刊合订本与特藏文献仅限馆内阅览,不外借。

## 二、续借

每册图书可续借 2 次,每次延长 30 天。

已被他人预约的图书不可续借。

续借须在到期日之前办理,可在图书馆网站或自助借还机上操作。

## 三、逾期

逾期归还的,每册每天罚款 0.5 元,罚款累计上限为该书定价。

逾期期间该读者不可再借新书。

## 四、预约与催还

预约图书到馆后为读者保留 3 天,逾期未取的视为放弃预约。

图书馆有权对已被他人预约的图书发出催还通知,读者须在收到通知后 7 天内归还。

## 五、遗失与损毁

图书遗失的,读者须购买相同版本图书赔偿,或按图书定价的 3 倍缴纳赔偿金。

书页污损、缺页的,按损毁程度赔偿,最低赔偿额为该书定价的 20%。

## 六、馆际互借

读者可通过馆际互借服务借阅合作馆藏,借期由出借馆规定,通常为 14 天。

馆际互借图书不可续借,逾期罚款标准为每册每天 1 元。

## 七、毕业与离校

毕业生办理离校手续前须还清所借图书并结清罚款,否则不予发放毕业证书。
"""

DORM_RULES = """宿舍管理条例

一、作息与门禁

宿舍楼门禁时间为每日 23:00。晚于门禁时间返回的,须在值班室登记并说明原因。

熄灯时间:周日至周四为 23:30,周五与周六为 24:00。

考试周期间熄灯时间统一延后至 24:00,由宿管中心提前一周公告。

二、电器使用

宿舍内禁止使用电饭锅、电磁炉、电热毯、热得快等大功率或发热类电器。

单个插座总功率不得超过 1000 瓦,禁止私拉电线或擅自改装线路。

违反者给予记过处分并没收违规电器。因违规用电引发事故的,
追究相应责任并赔偿损失。

三、访客与留宿

访客须在 22:00 前离开宿舍楼,并全程由被访学生在场陪同。

严禁留宿他人,包括本校其他宿舍的学生。留宿他人按违纪处理,给予警告处分。

四、卫生与检查

宿舍卫生检查每周一次,结果计入文明宿舍评比。

连续三次检查不合格的宿舍,全体成员取消本学年评优资格。

五、报修与退宿

设施损坏须通过宿管系统报修,紧急情况可直接联系值班室。

退宿须提前 7 天提交申请,并交还钥匙与门禁卡,经检查无损后办理手续。
"""

SCHOLARSHIP = """奖学金评定办法

一、总则

为激励学生勤奋学习、全面发展,学校设立国家奖学金、校级奖学金与专项奖学金三类。

奖学金评定遵循公开、公平、公正的原则,每学年评定一次。

二、国家奖学金

金额为每人每年 8000 元,名额由上级主管部门下达。

评定条件:上一学年 GPA 排名位于本专业前 10%,且无违纪记录、无不及格科目。

申请材料:个人申请书一份、成绩单一份、副教授及以上职称教师推荐信一封。

三、校级奖学金

一等校级奖学金每人每年 3000 元,GPA 排名须位于本专业前 30%。

二等校级奖学金每人每年 1500 元,GPA 排名须位于本专业前 50%。

校级奖学金与国家奖学金不可兼得,按金额高者发放。

四、专项奖学金

专项奖学金由社会团体或个人捐赠设立,金额与条件由捐赠协议约定,
通常为每人每年 2000 至 5000 元。

五、申请与评审

每学年 10 月 15 日为申请截止日期,逾期不再受理。

评审由院系初评、学校评审委员会终审两级组成。

评定结果于 11 月中旬公示,公示期为 5 个工作日。

六、发放与申诉

奖学金于公示结束后的次月一次性发放至学生本人账户。

对评定结果有异议的,可在公示期内向学生工作处提出书面申诉,
学生工作处须在 10 个工作日内予以答复。
"""

COURSE_HANDBOOK = """# Course Handbook

## 1. Credits

Each course carries 3 credits unless otherwise stated in the syllabus.

A bachelor's degree requires 120 credits in total to graduate.

A master's degree requires 30 credits, of which at least 18 must come from taught courses.

Credits earned from exchange programmes count towards the degree, up to a maximum of 24.

## 2. Course Load

The maximum course load is 18 credits per semester.

Students on academic probation are limited to 12 credits.

Students in their final semester may apply to reduce their load to 9 credits.

## 3. Adding and Dropping

Students may add a course within the first week of the semester.

Students may drop a course without penalty within the first two weeks.

After the second week, dropping a course results in a W grade on the transcript.

A W grade does not affect the GPA but remains visible to graduate admissions.

## 4. Attendance

Attendance is mandatory for laboratory and tutorial sessions.

Missing more than three laboratory sessions results in an automatic fail for that course.

Lecture attendance is not recorded, but lecturers may award up to 10 percent of
the final mark for participation.

## 5. Grading

The passing grade for undergraduate courses is D, equivalent to a GPA of 1.0.

A cumulative GPA of 2.0 or above is required to remain in good academic standing.

Students whose cumulative GPA falls below 2.0 for two consecutive semesters are
placed on academic probation.

## 6. Academic Integrity

All submitted work must be the student's own.

Plagiarism, contract cheating and unauthorised collaboration are reported to the
Academic Integrity Committee.

Confirmed misconduct results in a zero for the assessment and may lead to suspension.
"""

DINING_MENU = """Campus Dining Menu

Main Courses
Set Meal A - Grilled Chicken with Rice - $8.50
Set Meal B - Beef Noodles - $9.00
Set Meal C - Fried Rice with Egg - $7.50
Vegetarian Set - Tofu and Seasonal Greens - $7.50
Curry Chicken Rice - $8.00

Sides and Drinks
Soup of the Day - $2.00
Coffee - $2.50
Iced Lemon Tea - $3.00
Fresh Orange Juice - $3.50

Serving Hours
Breakfast 07:00 - 10:00
Lunch 11:00 - 14:00
Afternoon Tea 14:30 - 16:30
Dinner 17:00 - 20:00

Payment
The canteen accepts student cards only. Cash is not accepted.
Meal vouchers issued by the residential college can be used at the counter for
Set Meal A and the Vegetarian Set.
Top-up of the student card balance can be done at the kiosk next to the entrance.

Dietary Information
All vegetarian dishes are prepared in a separate kitchen area.
Nut-free options are marked with an asterisk on the daily board.
Students with food allergies should inform the counter staff before ordering.

Feedback
Comments and complaints can be submitted through the catering feedback form on
the student portal.
The catering manager reviews all submissions within 5 working days.
"""

LIBRARY_POLICY = """# Library Access Policy

## Opening Hours

The Main Library opens from 08:00 to 22:00 on weekdays.

On weekends the library opens at 10:00 and closes at 18:00.

During examination periods the library extends weekend closing time to 22:00.

The library is closed on public holidays.

## Study Spaces

Group study rooms may be booked for up to 2 hours per session, and up to 2 sessions per day.

Group study rooms must be booked online and require a minimum of three students to check in.

The silent study floor is located on level 4. Phone calls and group discussions are
not permitted on that floor.

Individual carrels on level 3 are available on a first-come basis and may not be reserved.

## Access and Membership

Current students and staff enter using their campus card.

Alumni may enter the library during weekdays only, and must register at the
reception desk with photo identification.

Visitors from partner institutions must be accompanied by a current staff member.

## Conduct

Food and hot drinks are not permitted inside the library. Covered water bottles are allowed.

Seats may not be reserved by leaving personal belongings for more than 30 minutes.

Library staff may remove unattended belongings from reserved seats and hold them
at the reception desk.

## Photography and Recording

Photography is permitted for personal study use only.

Recording lectures or events in library spaces requires prior written approval
from the Library Director.
"""

IT_POLICY = """Campus IT Usage Policy

1. Network Access

Students may register up to 3 devices on the campus WiFi network.
Each device must be registered with a valid student ID number.
Unregistered devices will be blocked automatically after 15 minutes.
Guest accounts are valid for 24 hours and must be sponsored by a current staff member.

2. Remote Access

A VPN connection is required to access library databases and internal file servers
from off campus.
The VPN client is available from the IT Services website.
Do not share VPN credentials with anyone, on or off campus.
VPN sessions are logged and inactive sessions are terminated after 8 hours.

3. Passwords

Passwords must be at least 12 characters and must be changed every 180 days.
Reusing any of the previous 5 passwords is not permitted.
Multi-factor authentication is required for administrative systems.

4. Acceptable Use

Campus network resources are provided for teaching, learning and research.
Peer-to-peer file sharing is prohibited on the campus network.
Running cryptocurrency mining software is strictly forbidden and will result in
immediate disconnection.

5. Data and Storage

Each student is allocated 50 GB of cloud storage.
Data stored on university systems may be subject to disclosure under public
records requests.
Students should not store sensitive personal data on shared drives.

6. Support

The IT Helpdesk is located on level 2 of the student centre.
It is open from 09:00 to 17:00 on weekdays and is closed on public holidays.
Urgent security incidents should be reported to security at any time.
"""

CORPUS = {
    "校园规章.md": CAMPUS_RULES,
    "图书馆借阅规则.md": LIBRARY_BORROWING,
    "宿舍管理条例.txt": DORM_RULES,
    "course_handbook.md": COURSE_HANDBOOK,
    "dining_menu.txt": DINING_MENU,
    "library_policy.md": LIBRARY_POLICY,
}
DOCX_DOCS = {"奖学金评定办法.docx": SCHOLARSHIP}
PDF_DOCS = {"it_policy.pdf": IT_POLICY}


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_docx(path: Path, content: str) -> None:
    import docx

    document = docx.Document()
    for raw_line in content.strip().splitlines():
        line = raw_line.strip()
        if line:
            document.add_paragraph(line)
    document.save(str(path))


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _write_pdf(path: Path, content: str, font_size: int = 10, leading: int = 14) -> None:
    """构造一个最小的单页 PDF,字体用内置 Helvetica(故只支持 ASCII)。"""
    lines = content.strip().splitlines()
    # A4 = 595 x 842 pt,留 50pt 边距
    body = ["BT", f"/F1 {font_size} Tf", f"{leading} TL", f"50 {842 - 50} Td"]
    for line in lines:
        body.append(f"({_escape_pdf_text(line)}) Tj T*")
    body.append("ET")
    stream = "\n".join(body).encode("ascii", errors="replace")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]

    buffer = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer += f"{number} 0 obj\n".encode() + payload + b"\nendobj\n"

    xref_offset = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode()
    buffer += b"0000000000 65535 f \n"
    for offset in offsets:
        buffer += f"{offset:010d} 00000 n \n".encode()
    buffer += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(buffer))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成内置示例语料")
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "data" / "raw"), help="输出目录"
    )
    parser.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    batch: list[tuple[str, str, object]] = []
    batch += [(n, c, _write_text) for n, c in CORPUS.items()]
    batch += [(n, c, _write_docx) for n, c in DOCX_DOCS.items()]
    batch += [(n, c, _write_pdf) for n, c in PDF_DOCS.items()]

    written, skipped = [], []
    for name, content, writer in batch:
        target = out / name
        if target.exists() and not args.force:
            skipped.append(name)
            continue
        writer(target, content)
        written.append(name)

    print(f"语料目录: {out}")
    print(f"已写入 {len(written)} 个文件: {', '.join(sorted(written)) or '(无)'}")
    if skipped:
        print(f"已跳过 {len(skipped)} 个已存在文件: {', '.join(sorted(skipped))}")
    print("已覆盖的解析格式: md / txt / docx / pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
