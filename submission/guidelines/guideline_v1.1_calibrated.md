# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603163 - Tạ Đức Thắng

> Chỉ ghi phần thay đổi so với v1.0. Mỗi quyết định khớp một dòng trong `qc/calibration_decisions.csv` (`case_id`).
> Cả 7 câu hỏi mở ở mục 6 của v1.0 đều đã chốt. Câu 1 chưa có ảnh ví dụ trong các ảnh đã duyệt nên chốt theo nguyên tắc, sẽ bổ sung ảnh khi gặp.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Có **bàn đạp** và **không có biển số** → `bicycle`. Có biển số, hoặc không có bàn đạp → `motorbike`. Không nhìn rõ → chọn lớp tin nhất + tick `can_review`. Sửa mục 2 (`bicycle`, `motorbike`). | chưa gặp trong các ảnh đã duyệt | CAL-08 |
| 2 | Khối xe máy đỗ dày đặc **gần camera**: từng xe hay `ignore_region`? | **Từng xe** nếu box cao ≥ 20 px và thấy ≥ 20% xe, kể cả bị xe phía trước che. Chỉ phần đuôi hàng ở xa (cao < 20 px, ≥ 5 xe sát nhau) mới gộp một `ignore_region`. Box máy cao < 20 px phải đổi thành `ignore_region` hoặc xoá, không giữ nguyên. Sửa mục 4.1.6. | `img_006` (hàng xe đỗ bên trái, gán từng xe), `img_014` (#155 xe máy cao 16 px: sai) | CAL-05 |
| 3 | Hàng nhô ra trên xe đạp / ba gác: box và mask có gồm hàng không? | Mui, thùng hàng **gắn liền** xe thuộc xe. Hàng chở trên xe chỉ tính vào box **và mask** khi nhô ra không quá 20% chiều dài xe (4.1.5 áp luôn cho mask). | `img_003` (xe ba gác chở rau, mui che cả khoang: box gồm mui + thùng) | CAL-09 |
| 4 | Người ngồi trong xe ba bánh có mui chỉ thấy thấp thoáng: ngưỡng 20% có hợp lý? | Giữ 20% nhưng thay ước lượng % bằng tiêu chí đếm được: thấy **đầu và một phần thân (vai / ngực)** → gán `person`; chỉ thấy tay / chân lẻ → không gán. Sửa mục 4.1.2 và dòng "Tuk-tuk có hành khách" ở mục 5. | `img_020` (khách ngồi trong tuk-tuk bên trái: thấy đầu + vai → phải có `person`; gold của tôi đã sót), `img_013` | CAL-03 |
| 5 | Kính chắn gió / cửa sổ xe có thuộc mask xe không? | **Kính thuộc mask xe**, kể cả nhìn xuyên thấy người bên trong. Riêng **khoang hở không có kính** của tuk-tuk / xích lô: phần nền / người nhìn xuyên qua bị khoét khỏi mask nếu lớn hơn 5% diện tích xe (áp 4.2.2). Người lái / khách là polygon `person` riêng, mask xe **không** chứa người (4.2.4). | `img_013` (mask SAM 2.1 nuốt người lái và lấn qua khoang hở: sai theo quy tắc này) | CAL-04 |
| 6 | Box máy chồng nhau quá nhiều: gộp hay giữ? | Hai box **cùng lớp trên cùng một vật** (IoU ≥ 0.5): giữ box ôm sát cả vật, xoá box còn lại. Hai **vật thật** chồng nhau (người trên xe, xe đỗ sát) giữ cả hai. Lô không có ảnh đêm; quy tắc áp cho mọi ảnh. Thêm vào mục 4.1. | `img_008`, `img_003` (YOLOE cho 2 box `three_wheeler` trên cùng một xe) | CAL-07 |
| 7 | Có mức `conf` tối thiểu để chấp nhận box máy không cần sửa? | **Không đặt.** `conf` chỉ để xếp thứ tự duyệt. Box điểm cao vẫn sai: `img_001` `truck` 0.88 (thực là SUV → `car`), `img_003` `three_wheeler` 0.61 (thực là xe tải). Sửa mục 3 (thuộc tính `conf`). | `img_001`, `img_003` | CAL-06 |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_012`, `img_020`, `img_013` | Tuk-tuk có mui nhìn từ bên cạnh / phía sau | YOLO26m gọi `car` (0.65–0.81 ở `img_012`, 0.42 ở `img_013`) hoặc `truck` (0.31 ở `img_020`) | Luôn `three_wheeler`. Định nghĩa v1.0 đã rõ, đây là lỗi máy: khi duyệt, soát riêng mọi box `car` / `truck` nằm trên xe ba bánh | Mục 5: thêm dòng ca khó (không đổi định nghĩa lớp). CAL-01 |
| `img_003` | Xe tải thùng kín ở xa, nhỏ, không đếm được bánh | YOLOE-26s gán `three_wheeler` (#9 0.61, #11 0.25) | `truck`. Dấu hiệu khi ở xa: **buồng lái + thùng hộp kín** = `truck`; `three_wheeler` cần thấy mui / khoang khách hoặc đếm được 3 bánh. Không chắc → lớp tin nhất + `can_review` | Mục 4.3: thêm dấu hiệu nhận biết. CAL-02 |
| `img_003`, `img_014` | Mảnh mép ảnh không nhận ra vật nào | YOLOE-26s gán `three_wheeler` điểm thấp (#12 0.16, x 969–1024) | Xoá. Vật cắt mép chỉ gán khi **nhận ra được** là vật thuộc lớp (4.1.3 không có nghĩa là giữ mọi box ở mép) | Mục 4.1.3: thêm câu "phải nhận ra được vật". CAL-02 |
| `img_013` | Mask SAM cho tuk-tuk có người lái ngồi bên trong khung xe | SAM 2.1 (3 click) nuốt người lái + lấn qua khoang hở; MobileSAM tách người tốt hơn nhưng vỡ 2 mảnh | Mask xe không chứa người; khoang hở > 5% khoét ra; cần thêm click âm lên đầu / chân người lái hoặc `--box` | Mục 4.2.2, 4.2.4. CAL-04 |
| `img_024`, `img_026`, `img_028`, `img_029`, `img_033` | Bẫy lọt: vật bị xoá khỏi pre-label, box bị nới lỏng | (bẫy do script cài) | Lỗi quy trình duyệt, không phải guideline mơ hồ → không đổi guideline; áp 3 quy tắc giảm thiên kiến ở `REPORT.md` mục 4 | Không đổi. CAL-10 |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt 7 câu hỏi mở (xe đạp điện; xe máy đỗ gần camera; hàng nhô ra cho cả mask; người trong xe ba bánh theo tiêu chí "đầu + vai"; kính thuộc mask, khoang hở khoét; gộp box trùng; không có ngưỡng `conf` tự chấp nhận). Thêm dấu hiệu `truck` vs `three_wheeler` khi ở xa, quy tắc mảnh mép ảnh, ca khó tuk-tuk bị gọi `car`. | CAL-01 → CAL-09; CAL-10 chỉ đổi quy trình duyệt |
