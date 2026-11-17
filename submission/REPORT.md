# Báo cáo Day 08 — 2A202603163 - Tạ Đức Thắng

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
| `img_013` (tuk-tuk hồng, có người lái) | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương: thân `400,480`, mui `600,110`; 1 âm: lưng người lái `720,300`) | **Có**: mask dính gần hết người lái (thân, chân) và một phần người ngồi sau xe nhìn qua cạnh hở | 317 255 px (45.4% ảnh), 1 polygon. Một click âm chỉ khoét được lưng áo, không tách được đầu và chân người lái; mask còn lấn qua khoang hở sang người ở quán phía sau | `tuktuk_img013_sam2.jpg` |
| `img_013` (cùng vật, cùng click: so sánh phiên bản) | SAM 1 (`mobile_sam.pt`) | 3 (giống dòng trên) | **Gần như không**: người lái và người phía sau nằm ngoài mask | 245 541 px (35.2% ảnh), **2 polygon**. Tách người tốt hơn SAM 2.1 nhưng hụt một mảng ở phần đầu xe / sàn trước, mask vỡ thành 2 mảnh | `tuktuk_img013_sam1.jpg` |
| `img_004` (tuk-tuk đỏ giữa hàng xe) | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương: thân `620,480`, đầu xe `700,560`; 1 âm: người đứng bên trái `465,450`) | Không: người đứng bên trái và xe xanh phía sau nằm ngoài mask | 93 190 px (13.3% ảnh), 1 polygon. Biên bám xe tốt; lem một dải **bóng đổ** dưới cản trước vì click `700,560` đặt sát mặt đất | `tuktuk_img004_sam2.jpg` |
| `img_017` (tuk-tuk hồng cạnh xe xanh) | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 dương: khoang khách `560,420`, thân `680,500`; 1 âm: `645,355`) | Không (khoang không có người) | 84 532 px (12.1% ảnh), 1 polygon. Mask sạch, đủ mui, thân, bánh; không dính xe xanh chạm phía sau. Click âm rơi đúng lên mui nhưng mui vẫn được giữ: click âm đặt sai chỗ, không có tác dụng | `tuktuk_img017_sam2.jpg` |

Ghi chú: các lần chạy Part 5 chạy trên **CPU** (máy cá nhân, ultralytics 8.4.215), khác các lần r01–r06 chạy trên GPU `cuda`; Part 5 không đo thời gian nên không so tốc độ.

- **Kết luận phiên bản:** giữ SAM 2.1 làm bản chính (mục 1). Trên xe không có người (`img_004`, `img_017`) SAM 2.1 cho mask liền một mảnh, bám biên tốt chỉ với 3 click. Nhưng ca khó nhất (`img_013`, người lái ngồi trong xe) cho thấy SAM 2.1 có xu hướng "nuốt" cả nhóm người + xe thành một vật; MobileSAM tách người tốt hơn nhưng mask hụt và vỡ mảnh. Không bản nào cho mask đúng contract chỉ với 3 click: cần thêm 1–2 click âm lên đầu / chân người lái hoặc thêm `--box` quanh xe.
- **Đối chiếu guideline mục 4.2:**
  - *Quy tắc giúp:* 4.2.3 (bóng đổ không thuộc vật) cho phép chỉ ra ngay lỗi ở `img_004`; 4.2.4 (hai vật chạm nhau không được chồng mask) cùng mục 4.1 quy tắc 4 (người trên xe là vật `person` riêng) xác định rõ mask SAM 2.1 ở `img_013` là **sai**, dù nhìn qua trông "đẹp".
  - *Quy tắc thiếu:* (1) 4.2.4 nói "vùng chạm thuộc vật ở phía trước", nhưng người lái ngồi **bên trong** khung xe (trước ghế, sau cột mui): chưa rõ phần nào thuộc xe, phần nào thuộc người. (2) 4.2.2 chỉ nói về lỗ thủng nhỏ / lớn, chưa nói khoang khách **hở hai bên** của tuk-tuk: nhìn xuyên qua thấy người và quán phía sau thì có khoét khỏi mask không (SAM 2.1 ở `img_013` lấn vào đúng vùng này). (3) Chưa có hướng dẫn đặt click: click sát mặt đất kéo theo bóng đổ (`img_004`), click âm đặt trên chính vật thì vô tác dụng (`img_017`). Ba điểm này đưa vào calibration v1.1.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: chưa thử (ưu tiên thời gian cho Part 4 và 6); toàn bộ mask Part 5 làm offline bằng `scripts/sam2_points.py`.

## 6. Calibration (Part 6)

- **Số quyết định: 10** (`calibration_decisions.csv`): 2 `loi_may` (CAL-01, CAL-02), 7 `cau_hoi_mo` (CAL-03 → CAL-09, chốt đủ 7 câu hỏi mở của v1.0), 1 `bay_lot` (CAL-10). Không có ca `can_review` vì lúc duyệt tôi không tick box nào (bản thân việc này là một dấu hiệu duyệt vội: ở `img_003` có 2 xe tải ở xa mà lẽ ra phải tick).
- **Quan trọng nhất:**
  - **CAL-02** (YOLOE bắn `three_wheeler` lên xe tải ở xa, mảnh mép ảnh, box trùng): đây là lỗi lặp lớn nhất của máy (20/42 box `three_wheeler` bị xoá, REPORT mục 2) và vẫn còn sót trong release (`img_003` #9, #11, #12 ở mẫu QC). Quy tắc mới: `three_wheeler` phải thấy mui / khoang khách hoặc đếm được 3 bánh; buồng lái + thùng hộp kín = `truck`.
  - **CAL-03** (người ngồi trong xe ba bánh): thay "thấy ≥ 20%" (không đo được bằng mắt) bằng tiêu chí đếm được "thấy đầu + vai". Chính tôi đã sót người như vậy ở gold `img_020`.
  - **CAL-06** (không có ngưỡng `conf` tự chấp nhận): `img_001` `truck` 0.88 và `img_003` `three_wheeler` 0.61 đều sai, nên điểm cao không thay được việc nhìn.
- **Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:** tiêu chí bàn đạp + biển số cho xe đạp điện (CAL-08); ngưỡng 20 px áp cả cho xe máy đỗ gần camera, box < 20 px phải đổi `ignore_region` hoặc xoá (CAL-05); ngưỡng hàng nhô 20% áp cho cả mask (CAL-09); "đầu + vai" cho người trong xe (CAL-03); kính thuộc mask, khoang hở > 5% khoét, mask xe không chứa người (CAL-04); gộp box trùng cùng lớp IoU ≥ 0.5 (CAL-07); không có ngưỡng `conf` (CAL-06); dấu hiệu `truck` vs `three_wheeler` khi ở xa và quy tắc mảnh mép ảnh (CAL-02); ca khó tuk-tuk bị gọi `car` (CAL-01). Mỗi quy tắc có ảnh ví dụ, trừ CAL-08 (chưa gặp xe đạp điện trong các ảnh đã duyệt).
- **Nguyên nhân gốc** (tách theo ca):
  - **Máy** (CAL-01, CAL-02): YOLO26 không có lớp xe ba bánh nên gọi `car` / `truck`; YOLOE ở ngưỡng 0.15 bắn lên mọi xe có thùng. Guideline đã rõ, không cần sửa định nghĩa, chỉ thêm dấu hiệu nhận biết.
  - **Guideline mơ hồ** (CAL-03 → CAL-09): 7 câu hỏi mở của v1.0 và 2 chỗ thiếu phát hiện ở Part 5 (khoang hở, người bên trong khung xe).
  - **Tôi chưa áp quy tắc** (CAL-10, và box < 20 px như `img_014` #155): quy tắc 4.1.6 đã có từ v1.0 nhưng tôi giữ nguyên box nhỏ của máy. Đây là lỗi quy trình duyệt, sửa bằng 3 quy tắc ở mục 4, không phải bằng guideline.

## 7. QC và quyết định (Part 6)

- **Mẫu:** 5 ảnh (`img_003`, `img_006`, `img_014`, `img_031`, `img_035`) = 0 ảnh `can_review` + 5 ngẫu nhiên, seed 42 (`qc_sample.py --n 5 --seed 42`), **115 đối tượng**, lấy từ release candidate 24 ảnh / 609 đối tượng (`build_release.py`). Người kiểm: Claude (AI) soát bằng mắt trên ảnh overlay có id và ảnh gốc phóng to, tôi xem lại phiếu (**hạn chế**: không phải người kiểm độc lập có chuyên môn; người duyệt lô là tôi).
- **Lỗi theo đối tượng: 6/115 = 5.2%** · **theo ảnh: 3/5 = 60.0%** (ngưỡng Release ≤ 2%). Hai ảnh sạch: `img_006`, `img_035`.
- **Top lỗi** (`qc_review.csv`):

  | Loại | Số | Ví dụ |
  |---|---|---|
  | sai lớp | 3/6 | `img_003` #9 và #11: xe tải thùng kín ở xa bị YOLOE gán `three_wheeler` (0.61, 0.25); `img_014` #155: xe máy cao 16 px, theo 4.1.6 phải là `ignore_region` hoặc xoá |
  | thừa | 2/6 | `img_031` #525: `person` 0.15 trên áo khoác treo bán; `img_003` #12: `three_wheeler` 0.16 ở mép phải, không có xe |
  | bỏ sót | 1/6 | `img_003`: xe đạp người phụ nữ đội nón đang dắt (#17) không có box `bicycle` |
  | sai vị trí / kích thước, sai thuộc tính, sai mask | 0 | (không kiểm được cạnh từng px ở ảnh dày; release không có mask) |

  5/6 lỗi là box **của máy được giữ nguyên** lúc duyệt (điểm 0.15–0.61), 4/6 lỗi rơi vào lớp `three_wheeler` / ngưỡng 20 px, đúng hai điểm yếu đã thấy ở mục 2 và 3. `sai_lop` chiếm 3/6 nên `qc_report.py` đánh dấu là **lỗi hệ thống**: cần sửa quy tắc / recipe (CAL-02, CAL-05), không chỉ sửa từng ảnh.
- **Giới hạn mẫu nhỏ:** 3/5 ảnh lỗi cho cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là **88%**. Ngay cả khi mẫu sạch 0/5, cận trên vẫn khoảng 3/5 = 60% (quy tắc 3), nên **mẫu 5 ảnh không bao giờ chứng minh được ngưỡng 2%**. Ở đây chiều ngược lại mới là kết luận chắc chắn: 3/5 ảnh lỗi là đủ để nói lô **chưa đạt**, dù tỉ lệ thật có thể thấp hơn 60%. Mẫu cũng chỉ có 115/609 = 19% đối tượng của release.
- **Script kiểm được / không kiểm được:** script đếm được trạng thái box, bẫy, tỉ lệ lỗi theo phiếu, box < 20 px (`validate_labels.py` báo 15 box). Script **không** biết box `three_wheeler` ở `img_003` thực ra là xe tải, không thấy xe đạp bị cả máy lẫn tôi cùng bỏ sót, không kiểm được box có ôm sát không, và không biết tôi có theo checklist thật hay không. Cả 6 lỗi QC đều chỉ thấy được bằng mắt.

**Quyết định: Rework**, ngày 2026-11-17. Lý do (đi theo Hold → Rework → Release):
1. *Hold?* Không: cả 7 câu hỏi mở đã chốt trong v1.1 (`qc_report.py` chạy với `--guideline-gap 0`), dữ liệu không có ảnh hỏng.
2. *Rework?* Có, đúng cả ba điều kiện: ảnh lỗi **3/5 = 60% > 2%**; một loại lỗi lặp (`sai_lop` **3/6 lỗi = 50%**, ≥ 3 lỗi); bắt bẫy **2/12 = 16.7% < 80%**. Chỉ một điều kiện là đủ Rework.
3. Không được Release: `batch_reviewed.json` hiện còn lỗi do chính quá trình duyệt (bẫy lọt 10/12, box máy sai được giữ nguyên).

**Việc cần làm tiếp (vòng 2):**
1. Sửa 6 lỗi QC trong CVAT (`img_003`: thêm `bicycle`, #9 #11 → `truck`, xoá #12; `img_014` #155 → `ignore_region` hoặc xoá; `img_031` xoá #525), export `submission/labels/qc_fixed.json`.
2. Duyệt lại **cả 24 ảnh** theo guideline v1.1 và 3 quy tắc ở mục 4 (quét lưới trước, ≥ 2.5 phút/ảnh, lượt lọc riêng `three_wheeler` / `truck` / `bicycle` / `bus`, áp 4.1.6 cho box < 20 px), tick `can_review` khi không chắc. Ưu tiên 6 ảnh "0 thay đổi" (`img_021`, `img_026`, `img_028`, `img_029`, `img_033`, `img_034`).
3. Vì đã biết đáp án 12 bẫy cũ, bẫy cũ **không còn đo được** thiên kiến ở vòng 2: cần bộ bẫy mới (seed khác) hoặc người kiểm độc lập.
4. Chạy lại `build_release.py`, `qc_sample.py --n 10 --force` (mẫu lớn hơn), `qc_report.py`; ghi kết quả vòng 2 vào báo cáo này.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** có, **Claude Code (Claude)** cho Part 5–6:
  - sửa ô 5.2 của notebook Colab (lỗi do dòng code bị ngắt khi dán) và chạy `sam2_points.py` cho Part 5 trên CPU máy tôi;
  - soát mẫu QC 5 ảnh (`qc_review.csv`), soạn `calibration_decisions.csv`, `guideline_v1.1_calibrated.md` và nháp `REPORT.md` mục 5–7.
  - *Kiểm tra:* tôi mở lại ảnh trong `submission/masks/` và `work/qc/review/` để đối chiếu từng lỗi; mọi số liệu lấy từ output script (`review_stats.md`, `qc_report_auto.md`), không có số tự đặt.
- **Gold có bị nhiễm không:** không. Gold 4 ảnh gán tay và commit (`0997a6a`, "Part 1") trước khi chạy model (`52f4ac4`, "Part 2-3"); AI chỉ được dùng từ Part 5.
- **Máy / GPU / CVAT:** Windows 11, laptop không dùng GPU cho model; model Part 2–4 chạy trên Google Colab (GPU T4, `device = cuda` trong `runs.csv`), Part 5 chạy trên CPU máy tôi. CVAT chạy bằng Docker trên máy. Vướng mắc: phiên Colab tự ngắt sau khi làm Part 3–4 trên máy, phải chạy lại ô cấu hình, nối lại Google Drive.
- **Dùng cache `data/cache/`?** Không: `y26_all` và `yoloe_all` (r05, r06) tự chạy trên Colab.
- **Lỗi / bug script đã gặp:** ô 5.2 của notebook Colab báo `SyntaxError: unterminated string literal` (dòng code dài bị ngắt khi dán), đã viết lại ô với các dòng ngắn; `merge_prelabels.py` không lọc box trùng **trong cùng một file** nên box `three_wheeler` trùng của YOLOE lọt sang batch (mục 2).
- **Một điều làm khác lần sau:** duyệt theo đúng checklist với sàn thời gian (≥ 2.5 phút/ảnh) và quét ảnh trước khi bật nhãn máy; tick `can_review` ngay khi không chắc (lần này 0 box được tick dù có ca như xe tải ở xa `img_003`); và nâng ngưỡng YOLOE lên 0.3 + lọc trùng trước khi gộp.
