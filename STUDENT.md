# Thông tin học viên

- Họ và tên: Tạ Đức Thắng
- MSHV: 2A202603163
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: Dthang-zz
- Repo bài nộp (URL): https://github.com/Dthang-zz/K5-L23-Day08-2A202603163-TaDucThang-AI-Assisted-Labeling
- Commit chốt bài (hash): Lab08
- Máy: Windows 11, không dùng GPU máy cho model (model chạy trên Google Colab GPU T4 CUDA; Part 5 chạy CPU)
- CVAT (Docker): v2.82.1

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 562/628 = 89.5% / 15/628 = 2.4% / 46/628 = 7.3% (mẫu số: 628 box máy); added: 15/592 = 2.5% (mẫu số: 592 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 54/82 = 65.9% / 54/62 = 87.1% (4 ảnh gold, in-sample nên lạc quan)
- Bắt bẫy: 2/12 = 16.7%
- Thời gian: gán tay 2.5 phút/ảnh (10 phút / 4 ảnh); duyệt có pre-label 1.25 phút/ảnh (30 phút / 24 ảnh)
- Lỗi QC: 60.0% theo ảnh (3/5), 5.2% theo đối tượng (6/115); cận trên 95% của tỉ lệ ảnh lỗi: 88%
- **Quyết định:** Rework — lý do một dòng: ảnh lỗi 3/5 > 2%, lỗi `sai_lop` lặp (3/6) và bắt bẫy 2/12 < 80%, nên lô phải duyệt lại theo guideline v1.1.
