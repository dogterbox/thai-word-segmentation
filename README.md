# Thai Word Segmentation using Deep Learning


*__Keywords__: Bi-LSTM, Classification, Deep learning, NLP*



### แนวคิด

ในภาษาไทยมีกฏการเขียนคำที่ค่อนข้างเคร่งคัด มีตัวอักษร (character, alphabet) บางตัวไม่สามารถใช้เขียนในตัวแหน่งเริ่มต้นของคำได้ เช่น สระอา (า) และในทางตรงกันข้ามมีตัวอักษรที่ใช้สำหรับเขียนขึ้นต้นของคำ เช่น สระแอ (แ) จากแนวคิดดังกล่าวการพัฒนาตัวตัดคำภาษาไทยในครั้งนี้ คือการพยายามที่จะเรียนรู้ตำแหน่งในการตัดคำว่าตัวอักษรดังกล่าวนั้นเป็นตัวอักษรตัวแรกของคำหรือไม่โดยพิจาณาบริบทจากคำก่อนหน้านี้ หรือกล่าวคือจำแนกประเภทของตัวอักษรในตำแหน่งนั้นๆ ว่าเป็นตัวอักษรขึ้นต้นของคำหรือไม่



### การรวบรวมข้อมูล

​         นำข้อมูลมาจากคลังข้อมูล [BEST Corpus](https://www.nectec.or.th/corpus/index.php?league=pm) ซึ่งมีการกำหนดขอบเขตของคำไว้เรียบร้อยแล้วจำนวน **506** ไฟล์ ประกอบด้วยข้อความของเอกสารจากหลายหมวดหมู่ ได้แก่ **บทความ สารานุกรม ข่าวและนวนิยาย**

```
บท|ความ|เกี่ยว|กับ|สื่อ|และ|สังคม|\nรู้|เท่า| |รู้|ทัน| |"|สื่อ|"|\n<NE>นิษฐา หรุ่นเกษม</NE>|\nนิสิต|ปริญญา|เอก| |:| |นิเทศศาสตร์|,| 
|<NE>จุฬาลงกรณ์มหาวิทยาลัย</NE>|\n(|บท|ความ|นี้|ยาว|ประมาณ| |4| |หน้า|กระดาษ| |A|4|)|\nเผยแพร่|ครั้ง|แรก|บน|เว็ปไซต์|
มหาวิทยาลัย|เที่ยงคืน| |วัน|ที่| |๑๗| |ธันวาคม| |๒๕๔๖|\nออก|ไป|อย่าง|กว้างขวาง| |นับ|จาก|สื่อ|แบบ|เก่า|ๆ| |ที่|เรา|คุ้นเคย| 
|เช่น| |หนังสือพิมพ์| |นิตยสาร| |โทรทัศน์| |และ|ภาพยนตร์| |สื่อ|บุคคล| |สื่อ|พื้นบ้าน| |หรือ|สื่อ|เฉพาะ|กิจ|เพื่อ|การ|ประชาสัมพันธ์|
|ไป|จน|ถึง|สื่อ|ใน|รูปแบบ|ที่|เรียก|กัน|ว่า|วัฒนธรรม|แบบ|ประชานิยม| |หรือ| |"|วัฒนธรรม|ป็อบ|"| |อย่าง|เช่น| |แฟชั่น|การ|แต่ง|กาย|
และ|ทรง|ผม| |ของ|เล่น|และ|ตุ๊กตา| |งาน|ฉลอง| |งาน|เทศกาล|ต่างๆ| |ฯลฯ| |ดัง|นั้น| |การ|ที่|จะ|เข้าใจ|ความ|หมาย| |รวม|ถึง|
|"|รู้|เท่า|ทัน|"| |อิทธิพล|จาก|สื่อ|ต่างๆ| |ทั้ง|แบบ|เก่า|และ|แบบ|ใหม่|เหล่า|นี้| |จำเป็น|จะ|ต้อง|มี|กรอบ|ความ|คิด|ที่|ชัดเจน| 
|เพื่อ|ให้|เห็น|ถึง|ปัจจัย|ต่างๆ| |ของ|สื่อ|แต่ละ|ประเภท|ที่|ซับซ้อน|และ|เกี่ยวพัน|ซึ่ง|กันและกัน|\nกรอบ|ความ|คิด|ดัง|กล่าว'
```



### การจัดเตรียมข้อมูล

1. **ลบคำที่ไม่เหมาะสม**ออก ได้แก่ **< AB>, < /AB>, < NE>, < /NE>, \n**
```
'บท|ความ|เกี่ยว|กับ|สื่อ|และ|สังคม|รู้|เท่า| |รู้|ทัน| |"|สื่อ|"|นิษฐา หรุ่นเกษม|นิสิต|ปริญญา|เอก| |:| |นิเทศศาสตร์|,| 
|จุฬาลงกรณ์มหาวิทยาลัย|(|บท|ความ|นี้|ยาว|ประมาณ| |4| |หน้า|กระดาษ| |a|4|)|เผยแพร่|ครั้ง|แรก|บน|เว็ปไซต์|มหาวิทยาลัย|
เที่ยงคืน| |วัน|ที่| |๑๗| |ธันวาคม| |๒๕๔๖|ออก|ไป|อย่าง|กว้างขวาง| |นับ|จาก|สื่อ|แบบ|เก่า|ๆ| |ที่|เรา|คุ้นเคย| |เช่น| 
|หนังสือพิมพ์| |นิตยสาร| |โทรทัศน์| |และ|ภาพยนตร์| |สื่อ|บุคคล| |สื่อ|พื้นบ้าน| |หรือ|สื่อ|เฉพาะ|กิจ|เพื่อ|การ|ประชาสัมพันธ์| 
|ไป|จน|ถึง|สื่อ|ใน|รูปแบบ|ที่|เรียก|กัน|ว่า|วัฒนธรรม|แบบ|ประชานิยม| |หรือ| |"|วัฒนธรรม|ป็อบ|"| |อย่าง|เช่น| |แฟชั่น|การ|
แต่ง|กาย|และ|ทรง|ผม| |ของ|เล่น|และ|ตุ๊กตา| |งาน|ฉลอง| |งาน|เทศกาล|ต่างๆ| |ฯลฯ| |ดัง|นั้น| |การ|ที่|จะ|เข้าใจ|ความ|หมาย|
|รวม|ถึง| |"|รู้|เท่า|ทัน|"| |อิทธิพล|จาก|สื่อ|ต่างๆ| |ทั้ง|แบบ|เก่า|และ|แบบ|ใหม่|เหล่า|นี้| |จำเป็น|จะ|ต้อง|มี|กรอบ|ความ|คิด|ที่|
ชัดเจน| |เพื่อ|ให้|เห็น|ถึง|ปัจจัย|ต่างๆ| |ของ|สื่อ|แต่ละ|ประเภท|ที่|ซับซ้อน|และ|เกี่ยวพัน|ซึ่ง|กันและกัน|กรอบ|ความ|คิด|ดัง|กล่าว
```
2. **สร้าง dictionary ของตัวอักษร** เพื่อใช้ในการแปลง character เป็น index มีทั้งหมด **180 character**  ซึ่งมีการเพิ่มตัวอักษรพิเศษ 2 ตัว คือ 
   1. < pad> เพื่อใช้ในการแทนที่ค่าว่าง 
   2. < unk> หรือ *unknow char* เพื่อใช้ในกรณีที่มีการนำ model ไปใช้งานต่อและทำการแปลง character เป็น index แต่พบว่ามี character ที่ไม่เคยปรากฏใน dataset ที่ใช้ในการ training มาก่อน

```
{'ก': 0,
 'ข': 1,
 'ค': 2,
 'ง': 3,
 ...
 ...
 ...
 '<pad>': 179
 '<unk>': 180}
```

*ตัวอย่างข้างต้น ไม่ได้แสดงตามค่า index จริง*



3. **แบ่งข้อมูล train-test-validation**

   Test        = แบ่งให้มูล *0.03 %* จากจำนวนไฟล์ทั้งหมด (506 ไฟล์)	--> 15  ไฟล์

   Train      = จำนวนไฟล์ทั้งหมด - จำนวนไฟล์ Test				   	     --> 477 ไฟล์

   Validate = แบ่งให้มูล *0.03 %* จากจำนวนไฟล์ Train			        --> 14  ไฟล์

4. **สร้าง dataset** ให้เหมาะสมกับ model โดยกำหนดให้มีการมองย้อนกลับไปพิจาณาข้อมูลก่อนหน้า (look back)      *เท่ากับ* *20* -- รูปแบบข้อมูล time sequence เพื่อใช้กับสถาปัตกรรมประเภท RNN กับการ**จำแนกประเภท** (classification) ว่า character ลำดับนั้นๆ เป็นอักษรขึ้นต้นหรือไม่ (No:0, Yes:1)

**ตัวอย่าง**: สร้าง dataset ของคำว่า **หิวข้าว** โดยกำหนดให้ look back = 4

```
"หิวข้าว"
look back = 4
```
```
--> |ห| ิ|ว|ข| ้|า|ว|
```
```
--> |<pad>|<pad>|<pad>|  ห  |  =  Yes     (X1, y1)
--> |<pad>|<pad>|  ห  |   ิ  |  =  No      (X2, y2)
--> |<pad>|  ห  |   ิ  |  ว  |  =  No      (X3, y3)
--> |  ห  |   ิ  |  ว  |  ข  |  =  Yes     (X4, y4)
--> |   ิ  |  ว  |  ข  |   ้  |  =  No       (X5, y5)
--> |  ว  |  ข  |   ้  |  า  |  =  No       (X6, y6)
--> |  ข  |   ้  |  า  |  ว  |  =  No       X7, y7)
```
```
X1 = [178, 178, 178, 135]   y1 = 1
X2 = [178, 178, 135, 144]   y2 = 0
X3 = [178, 135, 144, 131]   y3 = 0
X4 = [135, 144, 131, 95 ]   y4 = 1
X5 = [44,  131, 95,  160]   y5 = 0
X6 = [131, 95,  160  142]   y6 = 0
X7 = [95,  160  142, 131]   y7 = 0
```
```
....

X4 = [  [0, 0, 0, ..., 1, ..., 0],      <--- 135
        [0, 0, ..., 1, 0, ..., 0],      <--- 144
        [0, 0, 0, 0, ..., ..., 1]       <--- 131
        [0, 0, 1, ..., ..., 0, 0]  ]    <--- 92
		
y4 = [0, 1]                             <--- 1

....
```
จากข้อมูลของ [BEST Corpus](https://www.nectec.or.th/corpus/index.php?league=pm) เมื่อสร้างเป็น dataset จะได้ข้อมูลทั้งหมดดังนี้


```
Train set:     19,531,501
Validate set:  654,259
Test set:      583,377
รวม:            20,769,137
```



### สร้างโมเดล





![Model](https://github.com/dogterbox/thai-word-segmentation/blob/master/images/model.png?raw=true)



Layer

1. Input layer = data shape: (20 time step, 180 feature)
2. Hidden layer
   1. Bi-LSTM = unit: 90
   2. Dense = unit: 45 , activation: linear
3. Output layer
   1. Dense = num class (unit): 2, activation: softmax

Loss function = categorical-crossentropy  
Optimizer = rmsprop  
Metrics = accuracy  
Batch size = 1024  
Epoch = 20  
Step per epoch = 1,9073  

Callbacks

- Learning rate

  กำหนดให้มีค่าเริ่มต้นเท่ากับ **0.001** และปรับค่า learning rate ในทุกๆ epoch ให้ลดลงอย่างต่อเนื่อง (decay learning rate) และเท่ากับ **0.0001** เมื่อ train model ถึง epoch ที่ 20 พอดี

![Step decay](https://github.com/dogterbox/thai-word-segmentation/blob/master/images/step-decay.png?raw=true)

- Save model

  Save model เมื่อมีค่า accuracy จาก validate set ดีขึ้น




### ผลลัพธ์

##### 1. Model Loss - Model Accuracy

กราฟแสดงค่า loss และ accuracy ในแต่ละ epoch

- Loss

![Model loss](https://github.com/dogterbox/thai-word-segmentation/blob/master/images/history-loss.png?raw=true)

- Accuracy

![Model acc](https://github.com/dogterbox/thai-word-segmentation/blob/master/images/history-acc.png?raw=true)



##### 2. การวัดประสิทธิภาพ

```
Precision:  0.967
Recall:     0.981
Fscore:     0.974
Accuracy:   0.974
```



##### 3. ตัวอย่างการตัดคำ

```
บท|วิจารณ์| |-| |จาก|มุมมอง|ทาง|จริยธรรม|ส.ศิวรักษ์| |:| |ศูนย์บันเทิง|ครบ|วงจร|ของ|รัฐบาล|ทักษิณ|สุลักษณ์| |ศิวรักษ์| |-| 
|นัก|วิชาการ|พุทธศาสนา|บทความ|นี้|ยาว|ประมาณ| |10| |หน้า|กระดาษ| |a|4|เผยแพร่|ครั้ง|แรก|บท|เว็ปมหาวิทยาลัย|เที่ยงคืน| 
|วัน|ที่| |31| |มกราคม| |2547|ผล|กระทบ|นโยบาย|ศูนย์|บันเทิง|ครบ|วงจร|ของ|รัฐบาล|ทักษิณ|ต่อ|การ|พัฒนา|แก้ไข|ปัญหา|ความ|
ยากจน| |:| |จาก|มุมมอง|ทาง|จริยธรรม|-|๑|-|ประการ|แรก| |ขอ|ยืนยัน|ว่า|มุมมอง|หรือ|ทัศนะ|ทาง|จริยธรรม|นั้น| |ไม่|ได้|อยู่|ใน|
สายตา|ของ|นายก|รัฐมนตรี|ทักษิณ| ชินวัตร|เอา|เลย| |แม้|เขา|จะ|อ้าง|ว่า|เป็น|พุทธศาสนิก| |นั่น|เป็น|คำ|ลวง| |ที่|เขา|หลอก|ตน|
เอง|และ|มหาชน| |จน|เขา|เอง|ก็|เชื่อ|ตาม|คำ|ลวง|ของ|เขา| |นัก|การ|เมือง| |นัก|การ|ค้า|และ|นัก|โฆษณา|ชวน|เชื่อ|ทั้งหลาย|
|ชอบ|ใช้|ถ้อยคำ|ที่|เป็น|มุสาวาท|อยู่|เนืองนิตย์|จน|เชื่อ|ใน|ถ้อยคำ|นั้น|ๆ|เอา|เลย| |ดัง|คน|ที่|คุ้น|อยู่|กับ|ความ|รุนแรง| |ย่อม|ยาก|
ที่|จะ|เข้าใจ|ได้|ใน|หนทาง|ของ|สันติภาวะ| |หรือ|คน|ที่|คุ้นอยู่|กับ|ความ|โลภ| |ย่อม|ยาก|ที่|จะ|เข้าใจ|ได้|ใน|เรื่อง|ของ|สัน|โดษ|
พระภิกษุรูป|หนึ่ง|เขียน|บทความ|ลง|มติ|ชน| |ราย|วัน| |ฉบับ|วัน|ที่| |๔| |ธันวาคม| |๒๕๔๖| |อย่าง|น่า|รับฟัง|มาก| |หาก|ไม่|มี|
ปฏิกิริยา|ใด|ๆ|จาก|คน|ใน|รัฐบาล|ปัจจุบัน|เอา|เลย| |จึง|ขอ|นำ|เอา|มา|อ่าน|ให้|ฟัง|กัน|ดัง|ต่อ|ไป|นี้|พ.ต.ท.ทักษิณ| ชินวัตร| 
|นั้น|ประสบ|ความ|สำเร็จ|อย่าง|สูง|ตาม|กรอบ|คิด|และ|ทฤษฎี|เศรษฐศาสตร์|ทุนนิยม| |พิสูจน์|ได้|จาก|ผล|ประกอบ|การ|และ|โครงสร้าง|
ของ|กำไร|กิจการ|ที่|ท่าน|ริเริ่ม| |ก่อตั้ง| |และ|บริหาร| |ก่อน|การ|เข้า|รับ|ตำแหน่ง|หรือ|แสดง|บทบาท|ทาง|การ|เมือง| |กระทั่ง|สามารถ|
ใช้|ความ|สำเร็จ| |เหล่า|นั้น| |เป็น|บาท|ฐาน|เข้า|ยึด|กุม|อำนาจ|รัฐ|ด้วย|กลไก|การ|เลือกตั้ง|สำเร็จ|ใน|ที่สุด|อาจ|นับ|ได้|ว่า| |นี่|เป็น|
ประวัติศาสตร์|ความ|สำเร็จ|ของ|ฝ่าย|ทุน| |ที่|รวดเร็ว|และ|เบ็ดเสร็จ|เด็ดขาด|ยิ่ง| |โดย|ที่|มิ|ได้|ใช้|กลไก|กอง|ทัพ| |ระบบ|ราชการ| 
|หรือ|ทำ|การ|ปฏิวัติ|-|รัฐประหาร| |แต่|อย่างใด|คง|ปฏิเสธ|ได้|ยาก| |ว่า|อำนาจ|เบ็ดเสร็จ|เด็ดขาด|ทั้ง|ทาง|ตรง|และ|ทาง|อ้อม| 
|ผ่าน|กลไก|ต่างๆ| |ของ| |พ.ต.ท.ทักษิณ| ชินวัตร| |นั้น| |ณ| |วินาที|นี้| |ตลอดจน|ที่|จะ|เกิด|ขึ้น|หลัง|จาก|การ|เลือกตั้ง|ใน|อนาคต|
อัน|ใกล้| |ยัง|ไม่|มี|นายก|รัฐมนตรี|จาก|การ|เลือกตั้ง|คน|ใด|เคย|มี|หรือ|ทำได้|มาก่อน|ความ|สำเร็จ|ทาง|ธุรกิจ|ด้วย|ระยะ|เวลา|อัน|สั้น| 
|จน|สามารถ|มี|ทุน|ทรัพย์|ระดับ|หลาย|หมื่น|หลาย|แสน|ล้าน|บาท| |ด้วย|ระยะ|เวลา|เพียง|สอง|ทศวรรษ|เศษ| |ใน|ระบบ|ทุนนิยม| 
|อาจ|ถือ|ได้|ว่า|เป็น|เรื่อง| |"|ความ|สามารถ|"| |ใน|การ|แสวงหา|กำไร|สูง|สุด|แต่|ต้อง|ไม่|ลืม|ว่า| |"|กำไร|สูง|สุด|"| |นั้น| 
|แม้|จะ|ถูก|กฎหมาย| |(|หรือ|กฎหมาย|ยัง|เอา|ผิด|ไม่|ได้|?|)| |ก็|มิ|ใช่|จะ|หมาย|ความ|ว่า|กระทำ|ไป|ใน|ทิศทาง|เดียว|กับ| 
|"|ทำนอง|คลองธรรม|"|บ่อย|ครั้ง|ที่|ความ|สำเร็จ|ทาง|ธุรกิจ|การ|ค้า|แบบ|สัมปทาน|ผูกขาด| |หรือ|ซื้อ|ถูก|มาก|-|ขาย|แพง|มาก| 
|จะ|มี|ลักษณะ|มิจฉา|อาชีวะ| |คือ|การ|เลี้ยง|ชีพ|ผิด| |หา|เลี้ยงชีพ|ใน|ทาง|ทุจริต| |ผิด|วินัย| |หรือ|ผิด|ศีลธรรม|อยู่|เสมอ|กล่าว|
กัน|มา|แม้|ใน|ชาด|กว่า| |ความ|สำเร็จ|อัน|รวดเร็ว|และ|ยิ่งใหญ่| |เมื่อ|ผสาน|เข้า|กับ|อำนาจ|เบ็ดเสร็จ|เด็ดขาด|ด้วย|แล้ว| |หาก|ผู้|นั้น
|ไม่|มี|ธรรม| |ไม่|มี|การ|เจริญ|สติ| |หรือ|มี|การ|ภาวนา|เพียงพอ| |ก็|ง่าย|ที่|จะ|เกิด|ความ|สำคัญ|มั่น|หมาย|ใน|ตน| |ด้วย|อหังการ|
-|มมังการ| |มี|มิ|จฉาทิฐิ| |นำ|พา|ตน|เอง|และ|พวก|พ้อง|ออก|นอก|ลู่|นอก|ทาง|ธรรมยิ่ง|ๆ| |ขึ้น|ประวัติศาสตร์|อินเดีย| |หรือ|แม้|ใน|
พระไตรปิฎก|จึง|กล่าว|ไว้|ว่า| |"|เศรษฐี|"| |ครั้ง|พุทธกาล|นั้น| |ต้อง|ประกาศ|และ|ผ่าน|การ|รับรอง|จาก|สมาชิก|ใน|ชุมชน| |ว่า|เป็น|
|"|คน|ดี|"| |เพื่อ|มิ|ให้|คน|ชั่ว|หยาบแอบ|แฝง|เข้า|มา|ใช้| |"|วิชา|มาร|"| |ช่วง|ชิง|พื้นที่|และ|สิทธิ|ของ|สมาชิก|อื่น|นายก|รัฐมนตรี|
คน|ปัจจุบัน|กล่าว|ด้วย|ความ|ภาคภูมิใจ|อยู่|เสมอ|ว่า| |ตน|เชื่อมั่น|ใน|ตัว|เอง| |เป็น|คน| |"|คิด|นอก|กรอบ|"| |พร้อม|ที่|จะ|หา|
|"|ลู่ทาง|ใหม่|ๆ|"| |มา|แก้|ปัญหา|หลาย|ครั้ง|ที่| |พ.ต.ท.ทักษิณ| ชินวัตร| |กล่าว|กับ|สื่อ|มวล|ชน| |หรือ|กับ|ผู้|ฟัง|กลุ่ม|อื่น|ๆ| 
|ทำนอง|ว่า| |"|กฎหมาย|นั้น|เป็น|เพียง|เครื่องมือ| |มี|ได้|ก็|แก้ได้| |เพื่อ|เป้าหมาย|ที่|วาง|ไว้|"|จะ|ว่า|ไป|แล้ว| |ความ|ข้อ|นี้|ไม่|ใช่|
เรื่อง|แปลก| |โดย|เฉพาะ|อย่าง|ยิ่ง| |หาก|เรา|หลงลืม|คำ|กล่าว|ที่|ว่า| |"|กฎหมาย|ออก|โดย|ชน|ชั้น|ใด| |ย่อม|รับ|ใช้|ชน|ชั้นนั้น|"|
แต่|ตรรกะ|เช่น|นี้|ไม่|สามารถ|ใช้|ได้|กับ| |"|หลัก|ศาสนธรรม|"| |หรือ|เนื้อ|แท้|ใน|พระธรรม|คำ|สอน|ของ|ทุก|ศาสนา| |ยิ่ง|กับ|
|"|หลัก|ศีลธรรม|-|จริยธรรม|"| |อัน|เป็น|แก่น|แกน|ของ|วิถี|แห่ง|ความ|สงบ|ร่ม|เย็น|ของ|มนุษยชาติ|ด้วย|แล้ว| |แม้|ว่า|ถึง|ที่สุด|สิ่ง|ผิด|
กฎหมาย|จะ|ได้|รับ|การ|แก้|กฎหมาย|ให้|กลับ|เป็น|ถูกต้อง| |ก็|มิ|ใช่|จะ|ถูก|ทำนอง|คลองธรรมเสมอ|ไป|กรณี| |"|หวย|รัฐ|"| 
|หรือ|การ|จะ|ให้|มี|แหล่ง|บันเทิง|ครบ|วงจร|ซึ่ง|มี|บ่อน|การ|พนัน|อยู่|ใน|นั้น| |เป็นตัว|อย่าง|ที่|ชัดเจน|ที่สุด|ใน|ขณะ|นี้| |กล่าว|คือ| 
|แม้|รัฐบาล|จะ|ใช้|อำนาจ|บริหาร| |พรรค|รัฐบาล|ใช้|เสียง|ข้าง|มาก|ใน|สภา|มา|เปลี่ยน|ดำเป็น|ขาว| |เปลี่ยน|ผิด| |(|กฎหมาย|)| |ให้|
เป็น|ถูก|ได้|ตาม|อำเภอใจ| |แต่|รัฐบาล|หรือ| |นายก|รัฐมนตรี|ก็|ไม่|สามารถ|เปลี่ยน|กฎ|ศีลธรรม| |หรือ|เปลี่ยน|หลัก|ธรรม|คำ|สอน|ได้|
```

### Challenge 🎉💪❗🎇🎉🎉💪❗🎇🎉
```
กนกคนตลกชวนดวงกมลคนผอมรอชมภมรดมดอมดอกขจรสองคนชอบจอดรถตรงตรอกยอมทนอดนอนอดกรนรอยลภมรดมดอกหอมบนขอนตรงคลอง
มอญลมบนหวนสอบจนปอยผมปรกคอสองสมรสมพรคนจรพบสองอรชรสมพรปองสองสมรยอมลงคลองลอยคอมองสองอรชรมองอกมองคอมองผมมอง
จนสองคนฉงนสมพรบอกชวนสองคนถอนสมอลงชลลองวอนสองหนสองอรชรถอยหลบสมพรวอนจนพลพรรคสดสวยหมดสนกรกนกชวนดวงกมลชงนม
ผงรอชมภมรบนดอนฝนตกตลอดจนถนนปอนจอมปลวกตรงตรอกจอดรถถลอกปอกลงสองสมรมองนกปรอทจกมดจกปลวกจกหนอนลงคอสมพรคงลอย
คอลอยวนบอกสอพลอคนสวยผสมบทสวดของขอมคนหนอคนสมพรสวดวนจนอรชรสองคนฉงนฉงวยงวยงงคอตกยอมนอนลงบนบกสมพรยกซองผงทอง
ปลอมผสมลงนมชงของสองสมรสมพรถอนผมนวลลออสองคนปนผสมตอนหลอมรวมนมชงสมพรสวดบทขอมถอยวกวนหกหนขอวรรคตอนวอนผองชนจง
อวยพรสองดวงสมรรอดปลอดนรกคนคนจรหมอนสกปรกฝนตกจนจอมปลวกยวบลงมดปลวกหนอนออกซอกซอนลงผสมนมชงจนบทสวดหมดผลสมพร
คนสกปรกคงหลงยกนมชงซดลงคอรอครอบครองสองคนสวยปลวกมดหนอนอลวนซอกซอนจนสมพรปวดคองอลงหอนนอนครวญนอนหงอซมบนกอง
หนอนกองปลวกรอหมอตรวจลมฝนสงบลงผองปวงชนพลพรรคครบคนของสองอรชรยกพลสมทบชกถองหวดตบสมพรจนถดถอยตกตมจมลงคลอง
```
```
กนก|คน|ตลก|ชวน|ดวง|กมลคน|ผอม|รอชมภมรด|ม|ดอม|ดอก|ขจร|สอง|คน|ชอบ|จอด|รถ|ตรง|ตรอก|ยอม|ทน|อด|นอน|อด|กรนรอย|
ลภม|รด|มด|อก|หอมบน|ขอน|ตรง|คลอง|มอญ|ลม|บน|หวน|สอบ|จน|ปอย|ผม|ปรกคอสอง|สมรสมพรคนจร|พบ|สอง|อรชรสมพรป|
องสองสมรย|อม|ลง|คลอง|ลอย|คอมอง|สอง|อร|ชรม|อง|อก|มอง|คอม|อง|ผม|มอง|จน|สอง|คน|ฉงน|สมพรบอก|ชวน|สอง|คน|ถอน|
สมอลง|ชลลอง|วอน|สอง|หน|สอง|อรชรถอย|หลบ|สมพรวอนจน|พลพรรค|สดสวย|หมด|สนกรกนกชวน|ดวง|กมลชง|นม|ผง|รอ|ชมภม|
รบ|นด|อน|ฝน|ตก|ตลอดจน|ถนน|ปอน|จอม|ปลวก|ตรง|ตรอก|จอด|รถ|ถลอกปอก|ลง|สองสมรม|องนกปรอท|จก|มดจก|ปลวก|จก|หนอน|
ลง|คอ|สมพร|คง|ลอย|คอ|ลอยวน|บอก|สอพลอคนสวย|ผสม|บท|สวด|ของ|ขอม|คน|หนอ|คน|สมพร|สวด|วน|จน|อรชรสอง|คน|
ฉงนฉงวยงวยงงคอ|ตก|ยอม|นอน|ลง|บน|บก|สม|พรยก|ซอง|ผง|ทอง|ปลอม|ผสม|ลง|นม|ชง|ของ|สอง|สมรส|มพรถ|อนผม|นวล|
ลออสอง|คน|ปน|ผสม|ตอน|หลอม|รวม|นม|ชง|สมพร|สวด|บท|ขอ|มถอย|วก|วน|หก|หน|ขอ|วรรค|ตอน|วอน|ผองชน|จง|อวย|พร|
สอง|ดวง|สมรร|อด|ปลอด|นรก|คน|คน|จร|หมอน|สกปรก|ฝน|ตก|จน|จอม|ปลวก|ยวบ|ลง|มด|ปลวก|หนอน|ออกซอกซอน|ลง|ผสม|
นม|ชงจน|บท|สวด|หมด|ผล|สม|พร|คน|สกปรก|คง|หลง|ยก|นม|ชง|ซดลง|คอรอครอบครอง|สอง|คน|สวย|ปลวก|มด|หนอน|อลวน|
ซอกซอน|จน|สมพรปวด|คอง|อลง|หอนนอนครวญ|นอน|หงอซมบน|กอง|หนอน|กองปลวกรอ|หมอ|ตรวจ|ลม|ฝน|สงบ|ลง|ผอง|ปวงชน|
พล|พรรคครบคน|ของ|สอง|อรชรยก|พลสมทบ|ชก|ถอง|หวด|ตบสม|พร|จน|ถดถอย|ตกตม|จม|ลง|คลอง
```



### รายละเอียดไฟล์

1. thai-word-segmentation/notebooks/**create_char_dictionary.ipynb**

   สำหรับสร้าง dictionary ของ character

2. thai-word-segmentation/notebooks/**thai_word_segmentation.ipynb**

   สร้าง model ตัดคำ

3. thai-word-segmentation/notebooks/**tokenizer.py**

   นำ model ที่พัฒนาขึ้นไป implement เป็น module สำหรับเรียกใช้ได้โดยง่าย

4. thai-word-segmentation/notebooks/**usage.ipynb**

   การใช้งานตัวตัดคำ

