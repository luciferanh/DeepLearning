#Chương 4
## 4.1 Phân loại ảnh là gì?
Phân loại hình ảnh là nhiệm vụ gán nhãn cho một hình ảnh từ một nhóm danh mục được xác định trước. Thực tế, điều này có nghĩa là nhiệm vụ của chúng ta là phân tích hình ảnh đầu vào và trả về một nhãn phân loại hình ảnh. Nhãn luôn nằm trong một tập hợp các danh mục có thể được xác định trước.
Ví dụ: giả sử rằng tập hợp các danh mục có thể có của chúng tôi bao gồm:
Bộ dữ liệu=[chó,mèo,gấu trúc]
Sau đó, chúng tôi trình bày hình ảnh sau (Hình 4.1) với hệ thống phân loại của chúng tôi:

<center><img src='C:\Users\USER\Desktop\dog.png'></center>
<center>Hình 4.1: Mục tiêu của hệ thống phân loại ảnh là lấy ảnh đầu vào và gán nhãn dựa trên một tập hợp các loại được xác định trước.</center>

Mục tiêu của chúng tôi là lấy ảnh đầu vào ở trên để gắn nhãn dựa trên bộ dữ liệu của chúng tôi. Trong trường hợp này hình ảnh được gán nhãn là chó.
Hệ thống phân loại của chúng tôi có thể gán nhiều nhãn cho hình ảnh thông qua xác xuất chẳng hạn như trường hợp này là chó 95%, mèo 4%, gấu trúc 1%
Nói cách khác,với hình ảnh đầu vào của chúng tôi là W × H pixel với ba kênh, tương ứng là Đỏ, Xanh lục và Xanh lam, mục tiêu của chúng tôi là lấy hình ảnh W × H × 3 = N hình ảnh pixel  và tìm ra cách phân loại chính xác nội dung của bức hình.
### 4.1.1 Lưu ý về thuật ngữ
Khi thực hiện học máy và học sâu, chúng tôi có một tập dữ liệu mà chúng tôi đang cố gắng trích xuất đặc trưng từ đó. Mỗi ví dụ / mục trong tập dữ liệu (cho dù đó là dữ liệu hình ảnh, dữ liệu văn bản, dữ liệu âm thanh, v.v.) là một điểm dữ liệu.Do đó, một tập dữ liệu là một tập hợp các điểm dữ liệu (Hình 4.2).Mục tiêu của chúng tôi là áp dụng thuật toán học máy và học sâu để học các đặc trưng cơ bản trong tập dữ liệu, cho phép chúng tôi phân loại chính xác các điểm dữ liệu mà thuật toán của chúng tôi chưa gặp phải.Hãy dành thời gian bây giờ để tự làm quen với thuật ngữ này: 
1. Tập dữ liệu của chúng tôi là tập dữ liệu ảnh
2. item Do đó, mỗi hình ảnh là một điểm dữ liệu.
Tôi sẽ sử dụng thuật ngữ hình ảnh và điểm dữ liệu thay thế cho nhau trong suốt phần còn lại, vì vậy hãy ghi nhớ điều này ngay bây giờ
<center><img src='C:\Users\USER\Desktop\datapoint.png'></center>
<center>Hình 4.2: Tập dữ liệu (hình chữ nhật bên ngoài) là tập hợp các điểm dữ liệu (hình tròn)</center>

### 4.1.2 Khoảng cách về ngữ nghĩa
Hãy xem hai bức ảnh (trên cùng) trong Hình 4.3. Chúng ta sẽ thấy rất dễ khi nhận ra sự khác biệt giữa hai bức ảnh - rõ ràng có một con mèo ở bên trái và một con chó ở bên phải. Nhưng tất cả những gì máy tính nhìn thấy là hai ma trận pixel lớn (dưới cùng).
Những gì máy tính nhìn thấy là một ma trận pixel lớn cho nên chúng ta đi đến vấn đề về khoảng cách ngữ nghĩa. Khoảng cách ngữ nghĩa là sự khác biệt giữa cách con người nhận thức nội dung của hình ảnh so với cách hình ảnh có thể được biểu diễn theo cách máy tính có thể hiểu được quy trình.
Một lần nữa, xem xét nhanh hai bức ảnh trên có thể cho thấy sự khác biệt giữa hai loài động vật. Nhưng trên thực tế, máy tính không biết có động vật nào trong ảnh để bắt đầu. Để làm rõ điều này, hãy xem Hình 4.4, có một bức ảnh chụp một bãi biển yên tĩnh.
Chúng tôi có thể mô tả hình ảnh như sau:
* ***Không gian***: Bầu trời ở trên cùng của hình ảnh và cát / đại dương ở dưới cùng
* ***Màu sắc***: Bầu trời có màu xanh đậm, nước biển có màu xanh nhạt hơn bầu trời, trong khi cát có màu rám nắng.
* ***Kết cấu***: Bầu trời có mô hình tương đối đồng đều, trong khi cát rất thô

Làm cách nào để chúng ta mã hóa tất cả thông tin này theo cách mà máy tính có thể hiểu được? Câu trả lời là áp dụng tính năng trích xuất để định lượng nội dung của hình ảnh. Trích xuất đặc trưng là quá trình lấy hình ảnh đầu vào, áp dụng thuật toán và thu được vectơ đặc trưng (tức là danh sách các số) định lượng hình ảnh của chúng ta.
Để thực hiện quá trình này, chúng tôi có thể xem xét áp dụng các tính năng được thiết kế thủ công như HOG, LBPs hoặc các phương pháp tiếp cận “truyền thống” khác để định lượng hình ảnh. Một phương pháp khác, là áp dụng học sâu để tự động tìm hiểu một tập hợp các tính năng có thể được sử dụng để định lượng và cuối cùng là gắn nhãn nội dung của chính hình ảnh.
Tuy nhiên, nó không đơn giản như vậy. . . bởi vì một khi chúng tôi bắt đầu kiểm tra hình ảnh trong thế giới thực, chúng tôi phải đối mặt với rất nhiều thách thức
<center><img src='C:\Users\USER\Desktop\chomeo.png'></center>
<center>Hình 4.3</center>
(Trên cùng) Bộ não của chúng ta có thể nhìn thấy rõ ràng sự khác biệt giữa hình ảnh có con mèo và hình ảnh có chứa con chó. (Phần dưới) Tuy nhiên, tất cả những gì máy tính "nhìn thấy" là một ma trận lớn của các con số. Sự khác biệt giữa cách chúng ta cảm nhận một hình ảnh và cách hình ảnh được biểu diễn (ma trận các con số) được gọi là khoảng cách ngữ nghĩa.

### 4.1.3 thách thức
Nếu khoảng cách ngữ nghĩa không đủ là một vấn đề, chúng ta cũng phải **xử lý các yếu tố khác nhau [10]** trong cách một hình ảnh hoặc đối tượng xuất hiện. Hình 4.5 hiển thị hình ảnh trực quan về một số yếu tố biến đổi này.
Để bắt đầu, chúng ta có **sự thay đổi góc nhìn**, trong đó một đối tượng có thể được định hướng / xoay theo nhiều chiều liên quan đến cách đối tượng được chụp và chụp. Dù chúng ta chụp chiếc Raspberry Pi này ở góc độ nào, nó vẫn là một chiếc Raspberry Pi.
Chúng tôi cũng phải tính đến **điểm thay đổi dữ liệu**. Bạn đã bao giờ gọi một tách cà phê cao, lớn hoặc venti từ Starbucks chưa? Về mặt kỹ thuật, chúng đều giống nhau - một tách cà phê. Nhưng chúng đều có kích thước khác nhau của một tách cà phê. Hơn nữa, cùng một loại cà phê venti đó sẽ trông khác biệt đáng kể khi nó được chụp gần so với khi nó được chụp từ xa hơn. Các phương pháp phân loại hình ảnh của chúng tôi phải có thể chấp nhận được các loại biến thể tỷ lệ này.
Một trong những biến thể khó tính nhất là **biến dạng**. Đối với những bạn đã quen thuộc với bộ phim truyền hình Gumby, chúng ta có thể thấy nhân vật chính trong hình trên. Đúng như tên gọi của chương trình truyền hình, nhân vật này có tính đàn hồi, co giãn và có khả năng uốn éo cơ thể ở nhiều tư thế khác nhau. Chúng ta có thể xem những hình ảnh này của Gumby như một kiểu biến dạng vật thể - tất cả các hình ảnh đều chứa nhân vật Gumby; tuy nhiên, tất cả chúng đều khác biệt đáng kể với nhau.
<center><img src='C:\Users\USER\Desktop\bien.png'></center>
Hình 4.4: Khi mô tả nội dung của hình ảnh này, chúng ta có thể tập trung vào các từ truyền đạt bố cục không gian, màu sắc và kết cấu - điều này cũng đúng với các thuật toán thị giác máy tính.
đối tượng chúng ta muốn phân loại bị ẩn khỏi tầm nhìn trong ảnh (Hình 4.5). Ở bên trái chúng ta phải có một hình ảnh của một con chó. Và ở bên phải, chúng tôi có một bức ảnh của cùng một con chó, nhưng hãy chú ý cách con chó đang nghỉ ngơi bên dưới tấm bìa, bị khuất khỏi tầm nhìn của chúng tôi. Con chó vẫn hiển thị rõ ràng trong cả hai hình ảnh - cô ấy chỉ hiển thị trong một hình ảnh hơn hình ảnh kia. Các thuật toán phân loại hình ảnh vẫn có thể phát hiện và gắn nhãn sự hiện diện của con chó trong cả hai hình ảnh.

Cũng như thách thức như các biến dạng và sai khớp nói trên, chúng ta cũng cần xử lý những thay đổi về **độ chiếu sáng**. Hãy xem tách cà phê được chụp trong điều kiện ánh sáng tiêu chuẩn và ánh sáng yếu (Hình 4.5). Hình ảnh bên trái được chụp với ánh sáng tiêu chuẩn trên cao trong khi hình ảnh bên phải được chụp với rất ít ánh sáng. Chúng tôi vẫn đang kiểm tra chiếc cốc giống nhau - nhưng dựa trên điều kiện ánh sáng, chiếc cốc trông khác biệt đáng kể (rất đẹp khi đường nối các tông dọc của chiếc cốc có thể nhìn thấy rõ ràng trong điều kiện ánh sáng yếu, nhưng không phải là ánh sáng tiêu chuẩn)
Tiếp tục, chúng ta cũng phải tính đến **sự lộn xộn của nền**. Bạn đã từng chơi trò chơi Where’s Waldo? (Hoặc Where’s Wally? Cho độc giả quốc tế của chúng tôi.) Nếu vậy, thì bạn biết mục tiêu của trò chơi là tìm ra người bạn áo sọc trắng đỏ yêu thích của chúng ta. Tuy nhiên, những câu đố này không chỉ là một trò chơi giải trí dành cho trẻ em - chúng còn là sự thể hiện hoàn hảo cho sự lộn xộn trong bối cảnh. Những hình ảnh này cực kỳ "nhiễu" và có rất nhiều thứ xảy ra trong đó. Chúng tôi chỉ quan tâm đến một đối tượng cụ thể trong hình ảnh; tuy nhiên, do quá nhiều “nhiễu”, không dễ để chọn Waldo / Wally. Nếu điều đó không dễ dàng đối với chúng tôi, hãy tưởng tượng máy tính không có hiểu biết về ngữ nghĩa của hình ảnh sẽ khó khăn như thế nào!
Cuối cùng, chúng tôi có **biến thể nội bộ lớp**. Ví dụ kinh điển về sự thay đổi giữa các lớp trong thị giác máy tính đang cho thấy sự đa dạng hóa của các loại ghế. Từ những chiếc ghế thoải mái mà chúng tôi sử dụng để cuộn tròn và đọc sách, đến những chiếc ghế lót trên bàn bếp của chúng tôi để họp mặt gia đình, đến những chiếc ghế trang trí nghệ thuật cực kỳ hiện đại được tìm thấy trong những ngôi nhà danh giá, một chiếc ghế vẫn là một chiếc ghế - và các thuật toán phân loại hình ảnh của chúng tôi phải có thể phân loại tất cả các biến thể này một cách chính xác.
<center><img src='C:\Users\USER\Desktop\vatthe.png'></center>
Hình 4.5: Khi phát triển một hệ thống phân loại hình ảnh, chúng ta cần phải nhận thức được cách một vật thể có thể xuất hiện ở các góc nhìn khác nhau, điều kiện ánh sáng, khớp cắn, tỷ lệ, v.v.

Nếu bạn sử dụng phương pháp tiếp cận quá rộng, chẳng hạn như “Tôi muốn phân loại và phát hiện mọi đồ vật trong nhà bếp của mình”, (nơi có thể có hàng trăm đồ vật) thì hệ thống phân loại của bạn không thể hoạt động tốt trừ khi bạn có nhiều năm kinh nghiệm xây dựng bộ phân loại hình ảnh - và thậm chí sau đó, không có gì đảm bảo cho sự thành công của dự án.
Nhưng nếu bạn **định hình vấn đề của mình và thu hẹp phạm vi**, chẳng hạn như “Tôi chỉ muốn nhận dạng bếp và tủ lạnh”, thì hệ thống của bạn có **nhiều khả năng chính xác và hoạt động tốt hơn**, đặc biệt nếu đây là lần đầu tiên bạn làm việc với phân loại hình ảnh và học sâu.
Điểm mấu chốt ở đây là **luôn xem xét phạm vi của trình phân loại hình ảnh của bạn**. Mặc dù học sâu và Mạng lưới thần kinh hợp pháp đã chứng minh được sức mạnh đáng kể và sức mạnh phân loại dưới nhiều thách thức, bạn vẫn nên giữ cho phạm vi dự án của mình chặt chẽ và được xác định rõ ràng nhất có thể.
Hãy nhớ rằng ImageNet [42], tập dữ liệu điểm chuẩn trên thực tế cho các thuật toán phân loại hình ảnh, bao gồm 1.000 đối tượng mà chúng ta gặp phải trong cuộc sống hàng ngày của mình.
Học sâu **không phải là phép thuật**. Thay vào đó, học sâu giống như một chiếc cưa cuộn trong nhà để xe của bạn - mạnh mẽ và hữu ích khi được sử dụng đúng cách, nhưng nguy hiểm nếu sử dụng mà không có sự cân nhắc phù hợp. Trong suốt phần còn lại của cuốn sách này, tôi sẽ hướng dẫn bạn hành trình học sâu và giúp chỉ ra khi nào bạn nên sử dụng các công cụ điện này và khi nào bạn nên chuyển sang cách tiếp cận đơn giản hơn (hoặc đề cập đến nếu vấn đề không hợp lý với hình ảnh phân loại để giải quyết).
## 4.2 Cách học
Có ba loại học tập mà bạn có thể gặp phải trong sự nghiệp học máy và học sâu của mình: học có giám sát, học không giám sát và học bán giám sát.
### 4.2.1 Học có giám sát
Ví dụ về việc tạo hệ thống lọc thư rác này là một ví dụ về học có giám sát. Học có giám sát được cho là loại học máy được nghiên cứu và biết đến nhiều nhất. Với dữ liệu đào tạo của chúng tôi, một mô hình (hoặc “phân loại”) được tạo ra thông qua quá trình đào tạo trong đó các dự đoán được thực hiện trên dữ liệu đầu vào và sau đó được sửa khi dự đoán sai. Quá trình đào tạo này tiếp tục cho đến khi mô hình đạt được một số tiêu chí dừng mong muốn, chẳng hạn như tỷ lệ lỗi thấp hoặc số lần lặp lại đào tạo tối đa.
Các thuật toán học có giám sát phổ biến bao gồm Hồi quy logistic, Máy vectơ hỗ trợ (SVM) [43, 44], Rừng ngẫu nhiên [45] và Mạng thần kinh nhân tạo.
Trong bối cảnh phân loại hình ảnh, chúng tôi giả định rằng tập dữ liệu hình ảnh của chúng tôi bao gồm các hình ảnh cùng với nhãn lớp tương ứng của chúng mà chúng tôi có thể sử dụng để dạy cho bộ phân loại học máy của mình về mỗi danh mục “trông như thế nào”. Nếu bộ phân loại của chúng tôi đưa ra dự đoán không chính xác, chúng tôi có thể áp dụng các phương pháp để sửa lỗi của nó.
<center><img src='C:\Users\USER\Desktop\123.png'></center>
Bảng 4.1: Bảng dữ liệu chứa cả nhãn lớp (chó hoặc mèo) và vectơ đặc trưng cho mỗi điểm dữ liệu (giá trị trung bình và độ lệch chuẩn của từng kênh màu Đỏ, Xanh lục và Xanh lam, tương ứng). Đây là một ví dụ về nhiệm vụ phân loại có giám sát.

Sự khác biệt giữa học tập có giám sát, không giám sát và bán giám sát có thể được hiểu rõ nhất bằng cách xem ví dụ trong Bảng 4.1. Cột đầu tiên của bảng của chúng tôi là nhãn được liên kết với một hình ảnh cụ thể. Sáu cột còn lại tương ứng với vectơ đặc trưng của chúng tôi cho mỗi điểm dữ liệu - ở đây, chúng tôi đã chọn để định lượng nội dung hình ảnh của mình bằng cách tính toán giá trị trung bình và độ lệch chuẩn cho từng kênh màu RGB, tương ứng
Thuật toán học tập có giám sát của chúng tôi sẽ đưa ra dự đoán trên từng vectơ đặc điểm này và nếu nó đưa ra dự đoán không chính xác, chúng tôi sẽ cố gắng sửa nó bằng cách cho nó biết nhãn chính xác thực sự là gì. Sau đó, quá trình này sẽ tiếp tục cho đến khi đáp ứng được tiêu chí dừng mong muốn, chẳng hạn như độ chính xác, số lần lặp lại của quá trình học tập hoặc đơn giản là một khoảng thời gian tường tùy ý.
**Để giải thích sự khác biệt giữa học có giám sát, không giám sát và bán giám sát, tôi đã chọn sử dụng phương pháp dựa trên tính năng (tức là trung bình và độ lệch chuẩn của các kênh màu RGB) để định lượng nội dung của hình ảnh. Khi chúng tôi bắt đầu làm việc với Mạng nơ-ron hợp pháp, chúng tôi thực sự sẽ bỏ qua bước trích xuất đối tượng địa lý và sử dụng chính các cường độ pixel thô. Vì hình ảnh có thể là ma trận M × N lớn (và do đó không thể vừa khít với ví dụ bảng / bảng tính này), tôi đã sử dụng quy trình trích xuất tính năng để giúp hình dung sự khác biệt giữa các loại học tập.**

### 4.2.2 Học không giám sát
Ngược lại với học có giám sát, học không giám sát (đôi khi được gọi là học tự học) không có nhãn liên kết với dữ liệu đầu vào và do đó chúng tôi không thể sửa mô hình của mình nếu nó đưa ra dự đoán không chính xác.
Quay trở lại ví dụ bảng tính, việc chuyển đổi một vấn đề học tập có giám sát thành một vấn đề học tập không có giám sát cũng đơn giản như xóa cột “nhãn” (Bảng 4.2).
Học không giám sát đôi khi được coi là “chén thánh” của học máy và phân loại hình ảnh. Khi chúng tôi xem xét số lượng hình ảnh trên Flickr hoặc số lượng video trên YouTube, chúng tôi nhanh chóng nhận ra rằng có một lượng lớn dữ liệu chưa được gắn nhãn có sẵn trên internet. Nếu chúng tôi có thể yêu cầu thuật toán của mình tìm hiểu các mẫu từ dữ liệu không được gắn nhãn, thì chúng tôi sẽ không phải mất nhiều thời gian (và tiền bạc) để gắn nhãn hình ảnh cho các tác vụ được giám sát.
Hầu hết các thuật toán học không giám sát đều thành công nhất khi chúng ta có thể tìm hiểu cấu trúc cơ bản của tập dữ liệu và sau đó, áp dụng các tính năng đã học của chúng ta cho một vấn đề học có giám sát khi có quá ít dữ liệu được gắn nhãn để sử dụng.
Các thuật toán học máy cổ điển cho việc học không giám sát bao gồm  PCA, k-mean, mạng nơ-ron,...
<center><img src='C:\Users\USER\Desktop\1234.png'></center>
Bảng 4.2: Các thuật toán học không giám sát cố gắng học các mẫu cơ bản trong tập dữ liệu mà không có nhãn lớp. Trong ví dụ này, chúng tôi đã loại bỏ cột nhãn lớp, do đó biến nhiệm vụ này thành một vấn đề học tập không có giám sát.
<center><img src='C:\Users\USER\Desktop\12345.png'></center>
Bảng 4.3: Khi thực hiện học bán giám sát, chúng tôi chỉ có nhãn cho một tập hợp con của các hình ảnh / vectơ đặc trưng và phải cố gắng gắn nhãn các điểm dữ liệu khác để sử dụng chúng làm dữ liệu đào tạo bổ sung.
Học không giám sát là một lĩnh vực nghiên cứu cực kỳ tích cực và vẫn chưa có lời giải.

### 4.2.3 Học bán giám sát
Vì vậy, điều gì sẽ xảy ra nếu chúng ta chỉ có một số nhãn được liên kết với dữ liệu của mình và không có nhãn nào cho cái còn lại? Có cách nào chúng ta có thể áp dụng một số kết hợp giữa học có giám sát và không giám sát mà vẫn có thể phân loại từng điểm dữ liệu không? Hóa ra câu trả lời là có - chúng ta chỉ cần áp dụng phương pháp học bán giám sát.
Quay lại ví dụ về bảng tính của chúng tôi, giả sử chúng tôi chỉ có nhãn cho một phần nhỏ dữ liệu đầu vào của chúng tôi (Bảng 4.3). Thuật toán học tập bán giám sát của chúng tôi sẽ lấy các phần dữ liệu đã biết, phân tích chúng và cố gắng gắn nhãn từng điểm dữ liệu chưa được gắn nhãn để sử dụng làm dữ liệu đào tạo bổ sung. Quá trình này có thể lặp lại nhiều lần khi thuật toán bán giám sát học “cấu trúc” của dữ liệu để đưa ra các dự đoán chính xác hơn và tạo ra dữ liệu huấn luyện đáng tin cậy hơn.
Học tập bán giám sát đặc biệt hữu ích trong thị giác máy tính, nơi thường tốn thời gian (ít nhất là về giờ làm việc) để gắn nhãn từng hình ảnh trong bộ đào tạo của chúng tôi. Trong trường hợp chúng tôi chỉ đơn giản là không có thời gian hoặc nguồn lực để gắn nhãn cho từng hình ảnh riêng lẻ, chúng tôi chỉ có thể gắn nhãn một phần nhỏ dữ liệu của mình và sử dụng phương pháp học bán giám sát để gắn nhãn và phân loại các hình ảnh còn lại.
Các thuật toán học bán giám sát thường giao dịch các tập dữ liệu đầu vào có nhãn nhỏ hơn để giảm độ chính xác phân loại. Thông thường, việc đào tạo được gắn nhãn chính xác hơn cho một thuật toán học có giám sát, thì nó càng có thể đưa ra các dự đoán chính xác hơn (điều này đặc biệt đúng với các thuật toán học sâu).
Khi số lượng dữ liệu đào tạo giảm, độ chính xác chắc chắn sẽ bị ảnh hưởng. Học bán giám sát xem xét mối quan hệ này giữa độ chính xác và lượng dữ liệu và cố gắng giữ độ chính xác của phân loại trong giới hạn có thể chấp nhận được đồng thời giảm đáng kể lượng dữ liệu đào tạo cần thiết để xây dựng mô hình - kết quả cuối cùng là một bộ phân loại chính xác (nhưng thông thường không bằng chính xác như một bộ phân loại được giám sát) với ít nỗ lực và dữ liệu đào tạo hơn.
## 4.3 The Deep Learning Classification Pipeline
Dựa trên hai phần trước của chúng tôi về phân loại hình ảnh và các loại thuật toán học tập, bạn có thể bắt đầu cảm thấy hơi lúng túng với các thuật ngữ mới, cân nhắc và những gì có vẻ là một lượng biến thể không thể vượt qua trong việc xây dựng bộ phân loại hình ảnh, nhưng sự thật là rằng việc xây dựng một trình phân loại hình ảnh khá đơn giản, khi bạn đã hiểu quy trình.
Trong phần này, chúng tôi sẽ xem xét một sự thay đổi quan trọng trong tư duy mà bạn cần thực hiện khi làm việc với công nghệ máy học. Từ đó, tôi sẽ xem xét bốn bước xây dựng bộ phân loại hình ảnh dựa trên học sâu cũng như so sánh và đối chiếu giữa học máy dựa trên tính năng truyền thống với học sâu end-to-end.
### 4.3.1  A Shift in Mindset
Trước khi gặp bất cứ điều gì phức tạp, hãy bắt đầu với thứ mà tất cả chúng ta (rất có thể) đều quen thuộc: dãy Fibonacci.
Dãy Fibonacci là một dãy số trong đó số tiếp theo của dãy được tìm thấy bằng cách cộng hai số nguyên trước nó.
Ví dụ, cho dãy 0, 1, 1, số tiếp theo được tìm bằng cách thêm 1+1 = 2. Tương tự, cho 0, 1, 1, 2, số nguyên tiếp theo trong dãy là 1+2 = 3. Theo mẫu đó , các số đầu tiên trong dãy như sau: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
Tất nhiên, chúng ta cũng có thể xác định mẫu này trong một hàm Python (cực kỳ không được tối ưu hóa) bằng cách sử dụng đệ quy:
<center><img src='C:\Users\USER\Desktop\2.png'></center>
Sử dụng đoạn code này, chúng ta có thể tính số thứ n trong dãy bằng cách cung cấp giá trị n cho hàm fib. Ví dụ: hãy tính số thứ 7 trong dãy Fibonacci:
<center><img src='C:\Users\USER\Desktop\4.png'></center>
Như bạn có thể thấy, dãy Fibonacci rất đơn giản và là một ví dụ về nhóm hàm:
1. Chấp nhận một đầu vào, trả về một đầu ra.
2. Quy trình được xác định rõ ràng.
3. Đầu ra có thể dễ dàng xác minh tính đúng đắn.
4. đệ quy chính nó
Nói chung, bạn có thể đã viết hàng nghìn hàng nghìn hàm thủ tục như thế này trong cuộc sống của mình. Cho dù bạn đang tính toán một chuỗi Fibonacci, lấy dữ liệu từ cơ sở dữ liệu hay tính toán giá trị trung bình và độ lệch chuẩn từ một danh sách các số, các hàm này đều được xác định rõ ràng và có thể dễ dàng xác minh tính đúng đắn

***Thật không may, đây không phải là trường hợp cho học sâu và phân loại hình ảnh!***

Nhớ lại Phần 4.1.2, chúng ta đã xem các hình ảnh của một con mèo và một con chó, được mô phỏng lại trong Hình 4.6. Bây giờ, hãy tưởng tượng bạn đang cố gắng viết một hàm thủ tục không chỉ có thể cho biết sự khác biệt giữa hai bức ảnh này mà còn bất kỳ bức ảnh nào của một con mèo và một con chó. Bạn sẽ hoàn thành nhiệm vụ này như thế nào? Bạn có kiểm tra các giá trị pixel riêng lẻ ở các tọa độ (x, y) khác nhau không? Viết hàng trăm câu lệnh if / else? Và bạn sẽ duy trì và xác minh tính đúng đắn của một hệ thống dựa trên quy tắc khổng lồ như thế nào? Câu trả lời ngắn gọn là: bạn không làm như vậy.
<center><img src='C:\Users\USER\Desktop\23.png'></center>
Hình 4.6: Bạn có thể viết một phần mềm để nhận ra sự khác biệt giữa chó và mèo trong hình ảnh như thế nào? Bạn có kiểm tra các giá trị pixel riêng lẻ không? Thực hiện một cách tiếp cận dựa trên quy tắc? Cố gắng viết (và duy trì) hàng trăm câu lệnh if / else?

Không giống như mã hóa một thuật toán để tính toán dãy Fibonacci hoặc sắp xếp một danh sách các số, cách tạo một thuật toán để phân biệt sự khác biệt giữa ảnh mèo và chó là không trực quan hoặc rõ ràng. Do đó, thay vì cố gắng xây dựng một hệ thống dựa trên quy tắc để mô tả từng danh mục “trông như thế nào”, thay vào đó, chúng tôi có thể thực hiện phương pháp tiếp cận theo hướng dữ liệu bằng cách cung cấp các ví dụ về từng danh mục trông như thế nào và sau đó dạy thuật toán của chúng tôi nhận ra sự khác biệt giữa danh mục bằng cách sử dụng các ví dụ này.
Chúng tôi gọi những ví dụ này là tập dữ liệu đào tạo của chúng tôi về các hình ảnh được gắn nhãn, trong đó mỗi điểm dữ liệu trong tập dữ liệu đào tạo của chúng tôi bao gồm:
1. Ảnh
2. Nhãn / danh mục (tức là chó, mèo, gấu trúc, v.v.) của hình ảnh

Một lần nữa, điều quan trọng là mỗi hình ảnh này phải có nhãn được liên kết với chúng vì thuật toán học tập có giám sát của chúng tôi sẽ cần nhìn thấy các nhãn này để “tự dạy” cách nhận ra từng danh mục. Hãy ghi nhớ điều này, hãy tiếp tục và thực hiện bốn bước để xây dựng mô hình học sâu.

### 4.3.2 Bước 1 : Thu thập dữ liệu
Thành phần đầu tiên của việc xây dựng mạng học sâu là thu thập tập dữ liệu ban đầu của chúng tôi. Bản thân chúng ta cần các hình ảnh cũng như các nhãn liên kết với mỗi hình ảnh. Các nhãn này phải đến từ một nhóm danh mục hữu hạn, chẳng hạn như: danh mục = chó, mèo, gấu trúc.

Hơn nữa, số lượng hình ảnh cho mỗi danh mục phải gần như đồng đều (tức là số lượng ví dụ giống nhau cho mỗi danh mục). Nếu chúng ta có gấp đôi số lượng hình ảnh con mèo so với hình ảnh con chó và gấp năm lần số lượng hình ảnh gấu trúc so với hình ảnh con mèo, thì bộ phân loại của chúng tôi sẽ trở nên thiên vị một cách tự nhiên khi bổ sung quá nhiều vào các danh mục được đại diện nhiều này.

Mất cân bằng lớp học là một vấn đề phổ biến trong học máy và có một số cách để khắc phục nó. Hãy nhớ rằng phương pháp tốt nhất để tránh các vấn đề học tập do mất cân bằng lớp học là tránh hoàn toàn mất cân bằng lớp học. 

### 4.3.3 Bước 2: Xử lí Dataset
Bây giờ chúng ta sẽ chia Dataset thành 2 phần :
1. Tập dữ liệu train
2. Tập dữ liệu test

Bộ dữ liệu train được sử dụng để "học" từng danh mục trông như thế nào bằng cách đưa ra dự đoán trên dữ liệu đầu vào và sau đó tự sửa khi dự đoán sai. Sau khi đã được đào tạo, chúng tôi có thể đánh giá việc thực hiện trên một tập dữ liệu test.

**Điều cực kỳ quan trọng là bộ đào tạo và bộ kiểm tra độc lập với nhau và không chồng chéo!** Nếu bạn sử dụng bộ thử nghiệm như một phần của dữ liệu đào tạo, thì bộ phân loại của bạn có một lợi thế không công bằng vì nó đã xem các ví dụ thử nghiệm trước đó và “học hỏi” từ chúng. Thay vào đó, bạn phải giữ bộ kiểm tra này hoàn toàn tách biệt với quá trình đào tạo của bạn và **chỉ sử dụng nó để đánh giá mạng của bạn.**

Các kích thước phân chia phổ biến cho bộ đào tạo và kiểm tra bao gồm 66,6% 33,3%, 75% / 25% và 90% / 10%, tương ứng ở Hình 4.7:
<center><img src='C:\Users\USER\Desktop\3.png'></center>
<center>Hình 4.7: Các ví dụ về phân chia dữ liệu </center>

Trong thực tế, chúng ta cần kiểm tra một loạt các siêu tham số này và xác định bộ tham số hoạt động tốt nhất. Bạn có thể bị cám dỗ để sử dụng dữ liệu thử nghiệm của mình để điều chỉnh các giá trị này, nhưng một lần nữa, đây là điều quan trọng không thể bỏ qua! Bộ thử nghiệm chỉ được sử dụng để đánh giá hiệu suất mạng của bạn.

Thay vào đó, bạn nên tạo phần tách dữ liệu thứ ba được gọi là **validation set**. Tập hợp dữ liệu này (thông thường) đến từ dữ liệu huấn luyện và được sử dụng làm "dữ liệu thử nghiệm giả" để chúng tôi có thể điều chỉnh các siêu tham số của mình. Chỉ sau khi chúng tôi xác định được các giá trị siêu tham số bằng cách sử dụng **validation set**, chúng tôi mới chuyển sang thu thập kết quả chính xác cuối cùng trong dữ liệu thử nghiệm.

Chúng tôi **thường phân bổ khoảng 10-20% dữ liệu đào tạo để xác nhận**. Nếu việc chia nhỏ dữ liệu của bạn thành nhiều phần nghe có vẻ phức tạp thì thực tế không phải vậy. Như chúng ta sẽ thấy trong chương tiếp theo, nó khá đơn giản và có thể hoàn thành chỉ với một dòng mã nhờ thư viện scikit-learning.

### 4.3.4 Bước 3 : Đào tạo model của bạn
Với bộ hình ảnh đào tạo của chúng tôi, bây giờ chúng tôi có thể đào tạo mạng của mình. Mục tiêu ở đây là để mạng của chúng tôi tìm hiểu cách nhận ra từng danh mục trong dữ liệu được gắn nhãn của chúng tôi. Khi mô hình mắc lỗi, nó sẽ học hỏi từ sai lầm này và tự cải thiện.

Vậy, việc “học” thực tế hoạt động như thế nào? Nói chung, chúng tôi áp dụng một hình thức giảm dần độ dốc, như đã thảo luận trong Chương 9. Phần còn lại dành riêng để trình bày cách đào tạo mạng nơ-ron từ đầu, vì vậy chúng tôi sẽ hoãn thảo luận chi tiết về quá trình đào tạo cho đến lúc đó.

### 4.3.5 Bước 4 : Đánh giá
Cuối cùng, chúng ta cần đánh giá mạng lưới được đào tạo của mình. Đối với mỗi hình ảnh trong bộ thử nghiệm của chúng tôi, chúng tôi đưa chúng lên mạng và yêu cầu mạng dự đoán nhãn của hình ảnh đó là gì. Sau đó, chúng tôi lập bảng các dự đoán của mô hình cho một hình ảnh trong bộ thử nghiệm.

Cuối cùng, các dự đoán của mô hình này được so sánh với các nhãn xác thực từ bộ thử nghiệm của chúng tôi. Các nhãn xác thực cơ bản đại diện cho danh mục hình ảnh thực sự là như thế nào. Từ đó, chúng tôi có thể tính toán số lượng dự đoán mà trình phân loại của chúng tôi nhận được chính xác và tính toán các báo cáo tổng hợp như độ chính xác, thu hồi và độ đo f, được sử dụng để định lượng hiệu suất của toàn bộ mạng của chúng tôi.

### 4.3.6 Feature-based Learning versus Deep Learning for Image Classification

1285 / 5000
Translation results
Trong cách tiếp cận truyền thống dựa trên tính năng để phân loại hình ảnh, thực sự có một bước được chèn vào giữa Bước # 2 và Bước # 3 - bước này là trích xuất đối tượng địa lý. Trong giai đoạn này, chúng tôi áp dụng các thuật toán được thiết kế thủ công như HOG [32], LBPs [21], v.v. để định lượng nội dung của hình ảnh dựa trên một thành phần cụ thể của hình ảnh mà chúng tôi muốn mã hóa (tức là hình dạng, màu sắc, kết cấu). Với những tính năng này, sau đó chúng tôi tiến hành đào tạo bộ phân loại của mình và đánh giá nó.

Khi xây dựng Convolutional Neural Networks, chúng ta thực sự có thể bỏ qua bước trích xuất tính năng. Lý do cho điều này là vì CNN là mô hình end-to-end. Chúng tôi trình bày dữ liệu đầu vào (pixel) cho mạng. Sau đó, mạng sẽ học các bộ lọc bên trong các lớp ẩn của nó có thể được sử dụng để phân biệt giữa các lớp đối tượng. Đầu ra của mạng sau đó là một phân phối xác suất trên các nhãn lớp.

Một trong những khía cạnh thú vị của việc sử dụng CNN là chúng ta không còn phải bận tâm đến các tính năng được thiết kế thủ công nữa - thay vào đó chúng ta có thể cho phép mạng của mình tìm hiểu các tính năng. Tuy nhiên, sự đánh đổi này phải trả giá đắt. Đào tạo CNN có thể là một quá trình không hề nhỏ, vì vậy hãy chuẩn bị dành thời gian đáng kể để làm quen với kinh nghiệm và thực hiện nhiều thử nghiệm để xác định những gì hiệu quả và không hiệu quả. 

### 4.3.7 Điều gì sẽ xảy ra khi những dự đoán không chính xác?
Chắc chắn, bạn sẽ đào tạo một mạng học sâu trên bộ đào tạo của mình, đánh giá nó trên bộ thử nghiệm của bạn (nhận thấy rằng nó đạt được độ chính xác cao), và sau đó áp dụng nó cho các hình ảnh nằm ngoài cả bộ đào tạo và thử nghiệm của bạn - chỉ để thấy rằng mạng hoạt động kém.

Vấn đề này được gọi là tổng quát hóa, khả năng cho một mạng tổng quát hóa và dự đoán chính xác nhãn lớp của một hình ảnh không tồn tại như một phần của dữ liệu huấn luyện hoặc thử nghiệm của nó.

Khả năng tổng quát hóa của mạng thực sự là khía cạnh quan trọng nhất của nghiên cứu học sâu - nếu chúng ta có thể đào tạo các mạng có thể tổng quát hóa cho các bộ dữ liệu bên ngoài mà không cần đào tạo lại hoặc tinh chỉnh, chúng ta sẽ đạt được những bước tiến lớn trong học máy, cho phép mạng được sử dụng lại trong nhiều lĩnh vực khác nhau.

Khả năng tổng quát hóa của một mạng lưới sẽ được thảo luận nhiều lần, nhưng tôi muốn đưa ra chủ đề ngay bây giờ vì bạn chắc chắn sẽ gặp phải các vấn đề tổng quát hóa, đặc biệt là khi bạn học theo dây của học sâu. Thay vì thất vọng với việc mô hình của bạn không phân loại chính xác hình ảnh, hãy xem xét tập hợp các yếu tố thay đổi được đề cập ở trên. Tập dữ liệu đào tạo của bạn có phản ánh chính xác các ví dụ về các yếu tố biến đổi này không? Nếu không, bạn sẽ cần thu thập thêm dữ liệu đào tạo (và đọc phần còn lại để tìm hiểu các kỹ thuật khác để chống lại sự tổng quát hóa)

## 4.4 Kết luận
Trong chương này, chúng ta đã tìm hiểu phân loại hình ảnh là gì và tại sao máy tính lại thực hiện tốt nhiệm vụ khó khăn như vậy (ngay cả khi con người làm việc đó một cách trực quan mà dường như không cần nỗ lực). Sau đó, chúng tôi đã thảo luận về ba loại máy học chính, học có giám sát, học không giám sát, học bán giám sát - cuốn sách này chủ yếu tập trung vào học có giám sát, trong đó chúng tôi có cả các ví dụ đào tạo và nhãn lớp học được liên kết với chúng. Học tập bán giám sát và học tập không giám sát đều là những lĩnh vực nghiên cứu mở cho học sâu (và trong học máy nói chung).

Cuối cùng, chúng tôi đã xem xét bốn bước trong quy trình phân loại học sâu. Các bước này bao gồm thu thập tập dữ liệu, chia dữ liệu thành các bước đào tạo, thử nghiệm và xác thực, đào tạo mạng của bạn và cuối cùng là đánh giá mô hình của bạn.

Không giống như các phương pháp tiếp cận dựa trên tính năng truyền thống yêu cầu chúng ta sử dụng các thuật toán được tạo thủ công để trích xuất các tính năng từ một hình ảnh, các mô hình phân loại hình ảnh, chẳng hạn như Mạng thần kinh kết hợp, là các bộ phân loại end-to-end tìm hiểu nội bộ các tính năng có thể được sử dụng để phân biệt giữa các các lớp hình ảnh.