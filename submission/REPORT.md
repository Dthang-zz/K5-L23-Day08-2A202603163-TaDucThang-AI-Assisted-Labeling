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

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

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
