# Báo cáo Day 08 — <<MSHV - Họ và tên>>

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

Các lần chạy (khớp `work/runs.csv`, đều chạy trên GPU `cuda`):

| run_id | Model | Ảnh | Ngưỡng | Số box | Giây/ảnh | Quan sát chính |
|---|---|---|---|---:|---:|---|
| r01 | YOLO26m | 8 pilot | 0.25 | 215 | 1.19* | person 131, motorbike 55, car 17, truck 3, bicycle 8, bus 1. Tuk-tuk xanh ở `img_002` **không** có box xe nào (chỉ box `person` cho người ngồi trong xe); xe bán tải bên phải bị gọi `truck 0.73` |
| r02 | YOLO26m | 8 pilot | 0.5 | 139 | 0.34 | Giảm 76 box (-35%): person 131 → 87, motorbike 55 → 32, bicycle 8 → 3. Phần mất chủ yếu là người / xe máy nhỏ ở xa: ngưỡng cao làm **sót**, còn tuk-tuk vẫn không xuất hiện |
| r03 | YOLOE-26s, `three_wheeler` | 8 pilot | 0.25 | 8 | 0.17 | Bắt đúng tuk-tuk `img_002` (0.93) nhưng có 2 box thừa (0.37 trên xe bán tải, 0.40 trên cụm xe máy). Chỉ 3/8 ảnh có box; ở ngưỡng 0.5 còn 2 box |
| r04 | YOLO26m + SAM 2.1 (`sam2.1_b.pt`) | `img_004`, `img_013` | 0.3 | 16 box / 16 mask | 1.85 | `img_013`: tuk-tuk bị gọi `car 0.42`; SAM vẫn ra mask bám theo thân tuk-tuk (gồm cả mui) nhưng **nhãn vẫn sai lớp**: mask tốt không cứu được lớp sai |
| r05 | YOLO26m | 36 (all) | 0.05 | 1669 | 0.11 | Nguồn cho sweep / Part 4 |
| r06 | YOLOE-26s, `three_wheeler` | 36 (all) | 0.05 | 125 | 0.06 | Nguồn cho sweep / Part 4 |

\* r01 chậm hơn r02 (cùng model, cùng ảnh) nhiều khả năng vì là lần chạy đầu (nạp model, khởi động CUDA), không phản ánh tốc độ suy luận thật.

**Recipe chốt** (ngưỡng theo lớp lấy từ `qc/threshold_sweep.md`, mục tiêu recall 90%, quét trên 4 ảnh gold):

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id`) |
|---|---|---|---|
| person | YOLO26m | **0.15** | Ngưỡng cao nhất còn đạt recall ≥ 90%: recall 23/25 = 92%, precision 23/43 = 53%. Ở 0.3 recall rơi xuống 22/25 = 88%. r02 cho thấy tăng ngưỡng làm mất người nhỏ ở xa, mà bỏ box thừa khi review nhanh hơn vẽ box mới |
| bicycle | YOLO26m | **0.2** | Recall 5/5 = 100%, precision 5/7 = 71%. Lên 0.25 là mất 2/5 xe đạp (recall 60%). Chỉ 5 vật gold → ước lượng thô |
| motorbike | YOLO26m | 0.3 (mặc định `--conf`) | Không ngưỡng nào đạt recall 90% (tối đa 11/14 = 79%, 3 xe sót ở mọi ngưỡng). Recall phẳng từ 0.1 đến 0.35, nên 0.3 cho precision 11/11 = 100% (F1 0.88 cao nhất) mà không mất thêm xe. 3 xe sót là việc của bước review (thêm tay), không phải của ngưỡng |
| car | YOLO26m | 0.3 (mặc định) | Recall đứng yên 7/8 = 88% từ 0.1 đến 0.85: chiếc sót duy nhất là lỗi **sai lớp** (xem dưới), không phải điểm thấp. Precision ở 0.3 thấp (7/15 = 47%) vì phần lớn FP là tuk-tuk bị gọi `car`, mà các box này sẽ bị box `three_wheeler` của YOLOE thay khi gộp (IoU ≥ 0.6), nên không cần nâng ngưỡng `car` |
| bus | YOLO26m | 0.3 (mặc định) | Chỉ 1 vật gold (< 5): không đủ để chọn ngưỡng theo lớp |
| truck | YOLO26m | 0.3 (mặc định) | Chỉ 2 vật gold (< 5), và cả 2 thực ra là **box trùng do tôi gán gold sai** (xem mục giới hạn): sweep của `truck` (TP = 0 ở mọi ngưỡng) không dùng được |
| three_wheeler | YOLOE-26s, prompt `tuk-tuk`, `cyclo`, `cargo tricycle` (`taxonomy.json`) | **0.15** | Recall 7/7 = 100%, precision 7/9 = 78%. Lên 0.2 mất 1/7, lên 0.25 mất 2/7. Điểm YOLOE cho tuk-tuk thật thường thấp (0.19–0.36 ở `img_012`) nên phải dùng ngưỡng thấp; YOLO26 không có lớp này (r01, r04) |

- Lệnh `merge_prelabels.py` (YOLOE đứng **trước** để `three_wheeler` thắng `car`/`truck` của YOLO26 khi chồng nhau):
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json \
      --thr "person=0.15,bicycle=0.2,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" \
      --out work/prelabel/merged_review/pre.coco.json
  ```
  Kết quả (28 ảnh batch + gold): **729 box** (person 430, motorbike 154, car 65, three_wheeler 52, bicycle 17, truck 7, bus 4). Bị loại vì chồng lên `three_wheeler` của YOLOE (IoU ≥ 0.6): **11 `car` + 6 `truck`** của YOLO26, đúng là các tuk-tuk bị gọi sai lớp như ở r04 / `img_012`. Ngoài ra 2 `bicycle`, 2 `bus` bị loại vì chồng box ưu tiên cao hơn. Rồi cài bẫy: `inject_traps.py --student 2A202603163` → 12 bẫy (missing 3, loose 3, extra 3, wrong_class 3) trên 12 ảnh.
- **Phiên bản SAM: SAM 2.1 (`sam2.1_b.pt`)**. Lý do: là bản mặc định / khuyến nghị của lab, cân bằng chất lượng và tốc độ; máy có GPU CUDA nên bản `b` chạy 1.85 giây/ảnh (r04, gồm cả YOLO26 + 16 mask trên 2 ảnh) là chấp nhận được, không cần hạ xuống `mobile_sam.pt` (SAM 1, dành cho máy yếu). Không dùng SAM 3 vì là model gated (cần xin quyền, ~3.5 GB) và bài chỉ cần mask cho 3 vật bằng click / box, không cần mask hàng loạt từ chữ.
- **Lỗi không sửa được bằng đổi ngưỡng** — sai lớp do YOLO26 là detector danh mục cố định (80 lớp COCO, không có tuk-tuk / xích lô / ba gác):
  - `img_012`: cả 4 tuk-tuk đều bị gọi `car` với điểm **0.65–0.81** (IoU 0.92–0.99 với box gold). Muốn bỏ chúng bằng ngưỡng thì phải lên > 0.81, sát với điểm của ô tô thật (0.86–0.94 ở `img_012`, `img_020`); lên 0.9 thì recall `car` rơi còn 4/8. Quan trọng hơn: ngưỡng chỉ **xoá** được box, không **đổi lớp** được: tuk-tuk bị bỏ sẽ thành vật sót, vẫn phải vẽ tay `three_wheeler`. `img_020`: 2 tuk-tuk bị gọi `car` (0.39, 0.30), tuk-tuk nhỏ phía xa bị gọi `truck` (0.31). `img_013` (r04): tuk-tuk → `car 0.42`.
  - `img_001`: xe SUV/bán tải màu đen (guideline: bán tải, SUV là `car`) bị gọi `truck 0.88` → đây là chiếc `car` sót duy nhất trong sweep (recall `car` kẹt ở 7/8 với mọi ngưỡng).
  - Cách sửa: dùng model **open-vocabulary** (YOLOE với prompt chữ) cho `three_wheeler` và gộp với thứ tự ưu tiên YOLOE > YOLO26 (`--iou 0.6`), để box `three_wheeler` thay box `car`/`truck` chồng lên. Ở `img_012` YOLOE có box cho cả 4 tuk-tuk (0.19–0.36, đều ≥ 0.15) nên gộp sẽ sửa được. Lỗi `truck` ↔ `car` kiểu `img_001` thì gộp không sửa được → phải bắt ở bước review (kiểm lớp theo định nghĩa trong guideline).
- **Box điểm cao mà sai / điểm thấp mà đúng:**
  - Điểm cao mà sai: `img_001` `truck 0.88` (thực là SUV → `car`); `img_012` 4 box `car` 0.65–0.81 (thực là tuk-tuk); `img_036` 2 box `car` 0.65 và 0.61 kích thước chỉ ~16×13 px nằm trong vùng đông người (gold là `person` / `ignore_region`, và dưới mức tối thiểu 20 px); `img_002` (pilot) `truck 0.73` cho xe bán tải.
  - Điểm thấp mà đúng: `img_012` YOLOE `three_wheeler` 0.19 (IoU 0.84) và 0.24; `img_001` `person 0.29` (IoU 0.79) và `bicycle 0.21` (IoU 0.62); `img_036` `person 0.18`, `motorbike 0.12` (IoU 0.78), `bicycle 0.23`. Tức là điểm số **không phải** xác suất đúng: 0.88 vẫn có thể sai lớp, 0.12 vẫn có thể là vật thật.
  - Một ca "trông như sai" nhưng **gold sai**: `img_020` YOLO26 cho `person 0.88` ở khách ngồi trong tuk-tuk bên trái; gold của tôi không có box này (bỏ sót người ngồi trên xe, trái quy tắc "người ngồi trên xe vẫn có box riêng"). Sweep tính box này là FP nên precision `person` bị kéo thấp oan.
- **Giới hạn của việc chọn ngưỡng:**
  - Chọn ngưỡng và chấm máy trên **cùng 4 ảnh gold** → precision / recall ở ngưỡng đã chọn là con số **lạc quan** (ngưỡng đã được "khớp" theo đúng các vật này), chưa kiểm trên ảnh mới. 4 ảnh lại khá lệch: `img_012`, `img_020` là cảnh Thái nhiều tuk-tuk, `img_001`, `img_036` là cảnh Việt Nam nhiều xe máy.
  - Số vật mỗi lớp rất ít: bicycle 5, three_wheeler 7, car 8, motorbike 14, person 25; bus 1, truck 2. Một vật thay đổi là recall nhảy 12–20 điểm % → ngưỡng chỉ là ước lượng thô.
  - Gold có lỗi do tôi: **2 box trùng** — `img_012` một tuk-tuk được gán cả `three_wheeler` lẫn `truck` (cùng toạ độ ~[239, 58, 175, 211]), `img_020` chiếc xe buýt bên phải được gán cả `bus` lẫn `truck` ([953, 136, 71, 87]); cộng thêm người ngồi trong tuk-tuk `img_020` bị sót. Do đó 2 vật `truck` trong gold đều là ảo, và recall / precision `person`, `three_wheeler` cũng bị lệch. Các lỗi này sẽ ghi vào calibration (Part 6) thay vì sửa ngầm gold.

## 2. AI error profile (từ `qc/review_stats.md`)

Nguồn: `review_stats.py --pre submission/prelabel/pre_traps.coco.json --final submission/labels/batch_reviewed.json --gold ... --key ...`, 24 ảnh batch. Accepted nghĩa là box sau duyệt trùng box máy với IoU ≥ 0.9, cùng lớp. Script đã **loại 24 đối tượng quanh 12 bẫy** khỏi bảng này: bẫy là lỗi do script cài vào, không phải lỗi của máy, nên tính vào sẽ làm sai profile của máy.

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 562 | 89.5% | 628 box máy |
| edited | 15 | 2.4% | 628 box máy |
| deleted | 46 | 7.3% | 628 box máy |
| added | 15 | 2.5% | 592 đối tượng sau duyệt |

Trong deleted + added có 2 ca **đổi lớp** (box đúng chỗ, sai lớp: `img_018` three_wheeler → bus, `img_027` motorbike → three_wheeler), nên số vật máy **sót thật** trong batch ≈ 15 − 2 = 13.

| Lớp | Box máy | accepted | deleted | added |
|---|---:|---|---|---|
| person | 372 | 354 (95.2%) | 15 (4.0%) | 3 |
| motorbike | 137 | 131 (95.6%) | 2 (1.5%) | 6 |
| car | 53 | 49 (92.5%) | 4 (7.5%) | 1 |
| bicycle | 10 | 6 (60%) | 3 (30%) | 0 |
| truck | 6 | 4 (67%) | 2 (33%) | 0 |
| bus | 3 | 3 (100%) | 0 | 1 |
| **three_wheeler** | **42** | **15 (35.7%)** | **20 (47.6%)** | 4 |

- **Lớp máy tệ nhất: `three_wheeler` (YOLOE).** Gần một nửa box bị xoá (20/42), chỉ 15/42 giữ nguyên. Đối chiếu lại bằng IoU 18 box bị xoá, mình thấy ba kiểu lỗi:
  - **Box trùng trên cùng một xe** (5 box, chồng lên một box `three_wheeler` khác với IoU 0.36–0.49), ví dụ `img_003` (0.80), `img_008` (0.86), `img_017`, `img_023`. Kiểu này một box ôm cả xe, box kia chỉ ôm thân hoặc mui. `merge_prelabels.py` không lọc trùng **trong cùng một file**, nên các box này lọt sang batch.
  - **Mảnh ở mép ảnh** (5 box): `img_003`, `img_009`, `img_014`, `img_022` (x = 0) và `img_013` (sát mép phải, rộng 20 px).
  - **Box rơi vào người / xe máy / quầy hàng** (khoảng 8 box), ví dụ `img_014` 0.63 và 0.33, `img_005`, `img_035`.

  Điểm số không cứu được: có box bị xoá điểm 0.80–0.86. Đây là cái giá của ngưỡng 0.15 chọn để đạt recall 7/7 trên gold (mục 1): đổi lại precision rất thấp trên batch.
- Lớp **tốt nhất**: `person`, `motorbike`, `car` có accepted 92–96%. Nhưng `motorbike` có `added` cao nhất (6) và lọt 1 bẫy `missing` (`img_026`), nên recall thật có thể thấp hơn bảng trên (xem mục 3).
- **So với gold**: máy (pre-label sau gộp) đạt precision 54/82 = **65.9%** (KTC 95%: 55–75%), recall 54/62 = **87.1%** (KTC 95%: 77–93%). Mẫu số là 62 đối tượng gold (4 ảnh, không tính `ignore_region`). Hai con số này **lạc quan**, vì ngưỡng chọn trên chính 4 ảnh này. Chúng còn bị lệch vì gold có lỗi (2 box trùng `truck`, 1 người sót ở `img_020`, xem mục 1). Đáng chú ý: recall trên gold cho thấy máy sót khoảng **13%** vật (8/62), trong khi trên batch tôi chỉ vẽ thêm **2.5%** (15/592). Khoảng chênh này là dấu hiệu tôi **không tìm thấy** hết vật máy sót, không phải máy sót ít.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Bắt bẫy: 2/12 = 16.7%** (KTC 95%: 5–45%; ngưỡng đạt là 80%). Theo loại:

  | Loại | Bắt | Bẫy lọt (ảnh: chuyện gì xảy ra với box) |
  |---|---|---|
  | missing | 0/3 | `img_024` car (ở nửa dưới ảnh dày xe), `img_026` motorbike, `img_028` person sát mép phải: không vẽ thêm |
  | loose | 0/3 | `img_033` motorbike và `img_029` person: giữ nguyên box lệch (IoU với box đúng chỉ 0.55 và 0.62); `img_013` three_wheeler xa: đổi sang `ignore_region` nhưng **giữ nguyên box lỏng** |
  | extra | 1/3 | bắt được `img_004` (motorbike ảo đã xoá); lọt `img_027` và `img_009` (person ảo, giữ nguyên IoU 1.0) |
  | wrong_class | 1/3 | bắt được `img_010` (bicycle → motorbike, có chỉnh box); lọt `img_023` (car bị gọi truck) và `img_018` (person nhỏ 17×32 px bị gọi bicycle) |

- **Vì sao lọt, bỏ bước nào của checklist:**
  - **Bước 0 (ẩn nhãn máy, quét ảnh trước) và bước 1 (bỏ sót)**: bỏ qua. Cả 3 bẫy `missing` đều lọt, và 2/3 nằm ở vùng tôi ít nhìn (mép phải `img_028`, nửa dưới của ảnh 65 box `img_024`). Tôi chỉ nhìn chỗ có box, không quét chỗ **không có** box.
  - **Bước 4 (box lỏng)**: không làm. 0/3 bẫy `loose` bị bắt, và cả lô chỉ có 15 box được sửa (2.4%). Tôi không zoom để kiểm cạnh box. Ca `img_013` cho thấy rõ: tôi có chạm vào box (đổi lớp) nhưng vẫn không nhìn cạnh của nó.
  - **Bước 3 (sai nhãn) chỉ làm với lớp "lạ"**: tôi xoá nhiều `three_wheeler` vì đã biết YOLOE hay sai (mục 1), nhưng tin luôn `truck` / `bicycle` / `person` của YOLO26. Hai bẫy wrong_class lọt đều mang lớp "trông hợp lý" ở vị trí đó.
  - **Mệt / vội theo thứ tự ảnh**: từ `img_003` đến `img_018`, 11/11 ảnh có ít nhất một thay đổi. Từ `img_021` đến `img_035`, **6/13 ảnh giữ nguyên 100% box máy** (`img_021` 44 box, `img_026`, `img_028` 42, `img_029` 42, `img_033`, `img_034`: tổng 192 box không đụng tới). 4 trong 10 bẫy lọt nằm đúng ở các ảnh "0 thay đổi" này (`img_026`, `img_028`, `img_029`, `img_033`).
  - Cùng triệu chứng: ở các ảnh dày xe máy (`img_005`, `img_018`, `img_021`, `img_024`, `img_028`, `img_029`), guideline yêu cầu dùng `ignore_region` cho cụm < 20 px, nhưng tôi giữ nguyên box nhỏ của máy. `validate_labels.py` báo 15 box cao < 20 px (ví dụ `img_018` motorbike 14 px, `img_023` car 15 px). Tôi đã chấp nhận luôn **cách chia vật của máy** thay vì áp quy tắc của guideline.
- **Accepted cao mà bắt bẫy thấp nói gì:** accepted 89.5% **không** chứng minh máy tốt. Nó chủ yếu đo việc tôi **bấm giữ**: cùng một quy trình duyệt cho accepted 89.5% nhưng chỉ bắt 16.7% lỗi đã biết chắc có. Nếu tôi bỏ qua 10/12 lỗi cài sẵn, rất có thể tôi cũng bỏ qua một phần tương tự lỗi thật của máy nằm trong 562 box "accepted" và trong số vật sót. Đây đúng là **thiên kiến tự động hoá**: nhãn máy trông hợp lý nên mắt tôi xác nhận thay vì kiểm. Vì vậy `batch_reviewed.json` hiện **chưa đủ tin cậy** để làm nhãn cuối (sẽ xét ở mục 7).

## 4. Máy có đáng dùng không?

- **Thời gian** (`time_log.csv`): gán tay gold 10 phút / 4 ảnh = **2.5 phút/ảnh** (62 vật → khoảng 9.7 giây/vật). Duyệt có pre-label 30 phút / 24 ảnh = **1.25 phút/ảnh** (592 vật → khoảng 3 giây/vật). Tính theo vật, duyệt nhanh hơn khoảng **3 lần**.
- **Thiên lệch khi so thời gian:**
  - Tôi gán gold trước nên lúc duyệt đã quen CVAT, quen 8 lớp và kiểu cảnh phố, nên thời gian duyệt được lợi. Ngược lại, ảnh batch dày hơn (628 box máy / 24 = 26 box/ảnh, gold chỉ 15.5 vật/ảnh).
  - Quan trọng hơn: 1.25 phút/ảnh **thấp hơn một nửa** mức khoảng 3 phút/ảnh mà checklist khuyến nghị, và đi kèm bắt bẫy 16.7%. Nghĩa là phần lớn tốc độ đến từ **duyệt hời hợt**, không phải từ máy. Nếu duyệt đúng checklist (khoảng 3 phút/ảnh) thì duyệt sẽ **chậm hơn** gán tay tính theo ảnh, và chỉ còn nhanh hơn khoảng 1.4 lần tính theo vật (3 phút / 26 vật ≈ 7 giây/vật so với 9.7 giây/vật). Gold chỉ 4 ảnh, ghi chú "4 ảnh khá nhanh", nên số gán tay cũng là ước lượng thô.
- **Kết luận: có, nhưng chỉ cho một số lớp, và với điều kiện duyệt lại.**
  - **Dùng pre-label cho `person`, `motorbike`, `car`**: accepted 92–96%, deleted ≤ 7.5%, recall máy trên gold 87%. Sửa ít box nhanh hơn vẽ hàng trăm box người / xe máy từ đầu.
  - **`three_wheeler` (YOLOE ở ngưỡng 0.15) gần như không có lợi**: 20/42 box phải xoá, chỉ 15/42 giữ nguyên. Lần sau nên nâng ngưỡng (trên sweep, 0.3 giữ 5/7 với precision 5/6), lọc box trùng trong file YOLOE trước khi gộp, hoặc vẽ tay lớp này (sau duyệt chỉ có 26 box `three_wheeler` trong 24 ảnh).
  - **`bicycle`, `truck`, `bus`**: quá ít box (≤ 10) để kết luận.
  - Điều kiện: số đo trên chỉ có giá trị nếu người duyệt bắt được lỗi. Với bắt bẫy 2/12, lợi ích đo được hiện nay **chưa đáng tin**, và lô cần được duyệt lại.
- **Ba quy tắc giảm thiên kiến của tôi** (mỗi quy tắc kiểm tra được):
  1. **Quét trước, đếm trước.** Mỗi ảnh: ẩn nhãn máy, quét theo lưới 3×3 (gồm 4 mép ảnh), đếm số xe và người, rồi mới bật nhãn máy và so số box với số đếm. Lệch ≥ 2 thì tìm vật sót trước khi làm bất cứ việc gì khác. *Kiểm:* ghi số đếm vào cột `note` theo ảnh. Mục tiêu bắt bẫy `missing` 3/3, và `added`/đối tượng sau duyệt tiến gần tỉ lệ sót trên gold (khoảng 13%) thay vì 2.5%.
  2. **Không có ảnh "0 thay đổi" nếu chưa qua lượt 2, và có sàn thời gian.** Duyệt tối thiểu 2.5 phút/ảnh (ảnh > 40 box: 4 phút), nghỉ sau mỗi 8 ảnh, ghi `time_log` riêng từng nhóm 8 ảnh. Ảnh nào giữ 100% box máy phải được mở lại một lượt chỉ để zoom kiểm cạnh box (bước 4). *Kiểm:* `minutes/images` ≥ 2.5 trong `time_log.csv`; số ảnh 100% accepted (lần này 6/24) phải có ghi chú "đã duyệt lượt 2"; bắt bẫy `loose` ≥ 2/3.
  3. **Lượt lọc theo lớp cho lớp ít và lớp dễ nhầm.** Sau lượt chính, dùng bộ lọc nhãn của CVAT hiện **chỉ** `truck`, `bicycle`, `bus`, `three_wheeler` trên cả lô, rồi đối chiếu từng box với định nghĩa trong `taxonomy.json` (bán tải / SUV là `car`, người không phải `bicycle`). Đồng thời áp quy tắc < 20 px → `ignore_region`. *Kiểm:* bắt bẫy `wrong_class` 3/3, và `validate_labels.py` không còn cảnh báo box < 20 px (lần này 15).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
