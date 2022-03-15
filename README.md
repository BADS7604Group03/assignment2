# Assignment2
# Highlight

# Introduction
 ในการทำ Deep Learning  ปัญหาหลักที่เราจะพบเจอคือการ ความซับซ้อน มี Parameter (Weight) จำนวนมหาศาล ตั้งแต่การเริ่มต้นเทรน และการปรับจูนค่า Parameter จนจะได้ค่าดีในระดับหนึ่งในการใช้งานได้  แต่ทั้งนี้กระบวนการดังกล่าวอาจจำเป็นต้องประกอบไปด้วยทรัพยกรณ์อันมหาศาล  เช่น ข้อมูลที่ใช้ในการเทรน เวลา ความสามารถและความรวดเร็วในการคิดคำนวณ อีกทั้งเราไม่สามารถคาดเดาถึงความซับซ้อนในโครงสร้างของรูปภาพแต่ละชนิดได้  ทางเลือกหนึ่ง สำหรับการทำ Deep Learning สำหรับข้อมูลใหม่ที่เราต้องการใช้งาน เราเรียกว่า Transfer Learning 
 Transfer Learning คือ เทคนิคที่ช่วยลดเวลาการเทรนโมเดล Deep Learning ด้วยการนำบางส่วนของโมเดลที่เทรนเรียบร้อยแล้ว กับงานที่ใกล้เคียงกัน มาใช้เป็นส่วนหนึ่งของโมเดลใหม่ โดยส่วนใหญ่จะใช้วิธีนำโมเดล ConvNet ที่เทรนกับชุดข้อมูล Dataset ขนาดใหญ่ (เช่น ImageNet ที่มีข้อมูลตัวอย่างจำนวน 1.2 ล้านรูป ประกอบด้วย 1000 หมวดหมู่) มาเป็นโมเดลตั้งต้นเพื่อเทรนต่อ กับ Dataset ในงานเฉพาะทาง หรือ ใช้สกัด Feature สำหรับงานที่ต้องการออกมา
 
 ![image](https://user-images.githubusercontent.com/83268624/158397149-bfa85588-bcfd-46ca-aa1b-e24222a3b05f.png)

             ภาพ ของ  อ.ดร.ณัฐโชติ พรหมฤทธิ์ ภาควิชาคอมพิวเตอร์  คณะวิทยาศาสตร์ มหาวิทยาลัยศิลปากร

ในงานทำงานของกลุ่มเรา เราจะนำ transfer learning โดยการนำ Weight ของ Pretrain CNN Models เช่น   ResNet50 ซึ่งมาจากการพัฒนามาจากทีม Microsoft  ,Inception V3ซึ่งมาจากการพัฒนามาจากทีม GoogLeNet   และ NASNetMobile ที่พัฒนามาจาก Google brain team เพื่อมาเปรียบเทียบประสิทธิภาพความแม่นยำในการทำจำแนก บรรยากาศในร้านอาหาร  เมนูอาหาร และ อาหาร อ้างอิงมาจาก  https://life.wongnai.com/internship-image-classification-wongnai-a1dbc2890766  

# Data

# Training Strategy

# Result/Discussion/Conclusion

## NASNetMobile

### Freeze All Layers 

### Unfreeze Layer 752 - 769
เราได้นำ NASNetMobile พร้อมน้ำหนักที่ได้จากการ train กับข้อมูล ImageNet มาเป็นส่วน Feature Extractor ซึ่งโมเดลจะมีจำนวน Layer ทั้งหมด 769 Layer และจำนวน Parameter ทั้งหมด 4,269,716 ตัว ทั้งนี้เราจะทำการ unfreeze Weight ใน Layer ตั้งแต่ 752 เป็นต้นไป เพื่อให้โมเดลปรับจูนน้ำหนักให้เหมาะสมกับข้อมูลในระหว่างที่ Train ได้ ซึ่งเราหวังว่าจะได้ model ที่มีค่า accuracy/loss ที่ดีขึ้น เมื่อเรานำมารวมกับส่วน Classification Layer จะมีจำนวน Layer ทั้งหมด 775 Layer และจำนวน Parameter ทั้งหมด 17,780,119 ตัว ทางทีมได้ train โมเดลผ่าน Google-Colab (gpu : Tesla V100-SXM2-16GB)  โดยผลจากการ train ได้ผลลัพธ์ดีที่สุดมี loss อยู่ที่ 0.233, accuracy อยู่ที่ 0.942 ( atmosphere : 0.90, food : 0.953, menu : 0.973) และมี roc อยู่ที่ 0.99 (atmosphere : 0.99, food : 0.99, menu : 0.99)

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/158398009-6833deb5-af66-4bd0-b812-785b8b6981fa.png">
  </br>(ภาพ accuracy และ loss ในระหว่างที่ train)

  <img width="800" src="https://user-images.githubusercontent.com/87576892/158398116-0e2f2293-8dd1-402e-8ba0-87f8c8a35f15.png">
  </br>(ภาพการทำ Grad-CAM เพื่อตรวจสอบโมเดล)
</p>

## ResNet50

### Freeze All Layers

### Unfreeze


## Inception V3

### Freeze All Layers

### Unfreeze

# Referance
- https://life.wongnai.com/internship-image-classification-wongnai-a1dbc2890766  

# Member 
1) ณัฐภณ อัศวเหม 6310422052 (% contribution in this homework: 16.67%)
3) ดวงธิดา แซ่แต้ 6310422056 (% contribution in this homework: 16.67%)
4) เมธี ประเสริฐกิจพันธุ์ 6310422053 (% contribution in this homework: 16.67%)
5) พีรพัทธ ตั้งไพบูลย 6310422024 (% contribution in this homework: 16.67%)
6) วิชิต ชำนาญนาวา 6310422055 (% contribution in this homework: 16.67%)
7) ไตรทิพย์ ศุภศิริวัฒนา 6310422009 (% contribution in this homework: 16.67%)


