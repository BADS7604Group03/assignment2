# Assignment2
# Highlight

# Introduction
 ในการทำ Deep Learning  ปัญหาหลักที่เราจะพบเจอคือการ ความซับซ้อน มี Parameter (Weight) จำนวนมหาศาล ตั้งแต่การเริ่มต้นเทรน และการปรับจูนค่า Parameter จนจะได้ค่าดีในระดับหนึ่งในการใช้งานได้  แต่ทั้งนี้กระบวนการดังกล่าวอาจจำเป็นต้องประกอบไปด้วยทรัพยกรณ์อันมหาศาล  เช่น ข้อมูลที่ใช้ในการเทรน เวลา ความสามารถและความรวดเร็วในการคิดคำนวณ อีกทั้งเราไม่สามารถคาดเดาถึงความซับซ้อนในโครงสร้างของรูปภาพแต่ละชนิดได้  ทางเลือกหนึ่ง สำหรับการทำ Deep Learning สำหรับข้อมูลใหม่ที่เราต้องการใช้งาน เราเรียกว่า Transfer Learning 
 Transfer Learning คือ เทคนิคที่ช่วยลดเวลาการเทรนโมเดล Deep Learning ด้วยการนำบางส่วนของโมเดลที่เทรนเรียบร้อยแล้ว กับงานที่ใกล้เคียงกัน มาใช้เป็นส่วนหนึ่งของโมเดลใหม่ โดยส่วนใหญ่จะใช้วิธีนำโมเดล ConvNet ที่เทรนกับชุดข้อมูล Dataset ขนาดใหญ่ (เช่น ImageNet ที่มีข้อมูลตัวอย่างจำนวน 1.2 ล้านรูป ประกอบด้วย 1000 หมวดหมู่) มาเป็นโมเดลตั้งต้นเพื่อเทรนต่อ กับ Dataset ในงานเฉพาะทาง หรือ ใช้สกัด Feature สำหรับงานที่ต้องการออกมา
 
 ![image](https://user-images.githubusercontent.com/83268624/158397149-bfa85588-bcfd-46ca-aa1b-e24222a3b05f.png)

             ภาพ ของ  อ.ดร.ณัฐโชติ พรหมฤทธิ์ ภาควิชาคอมพิวเตอร์  คณะวิทยาศาสตร์ มหาวิทยาลัยศิลปากร

ในงานทำงานของกลุ่มเรา เราจะนำ transfer learning โดยการนำ Weight ของ Pretrain CNN Models เช่น   ResNet50 ซึ่งมาจากการพัฒนามาจากทีม Microsoft  ,Inception V3ซึ่งมาจากการพัฒนามาจากทีม GoogLeNet   และ NASNetMobile ที่พัฒนามาจาก Google brain team เพื่อมาเปรียบเทียบประสิทธิภาพความแม่นยำในการทำจำแนก บรรยากาศในร้านอาหาร  เมนูอาหาร และ อาหาร อ้างอิงมาจาก  https://life.wongnai.com/internship-image-classification-wongnai-a1dbc2890766  

# Data
เนื่องจากทางทีมวางแผนจะใช้วิธี Transfer Learning เข้ามาช่วยในการพัฒนาโมเดลจำแนกประเภทรูปภาพทั้ง 3 คลาส 1 - บรรยากาศ(atmosphere), 2-อาหาร(food) และ 3-menu ทำให้สามารถรลดปริมาณรูปที่จะใช้ในการพัฒนาโมเดลได้ค่อนข้างเยอะ โดยทีมงานจะทำการเก็บรวบรวมรูปภาพที่จะใช้ในการ Train รูปภาพทั้ง 3 คลาส คลาสสละ 200 รูป รวมทั้งหมด 600 รูป จากการไปถ่ายจากร้านอาหาร และรวบรวมจากเว็บไซต์ต่างๆ โดยหลังจากเก็บรวบรวมรูปทั้งหมดเรียบร้อยแล้ว เพื่อให้สามารถนำมาใช้ในขั้นตอนต่อไปได้ง่าย จึงนำรูปทั้งหมดมาเตรียมไว้ให้อยู่ในรูป numpy ทั้มหมด 3 ไฟล์ 
1) x_image : เก็บข้อมูลรูปภาพ โดยจะมีขนาด (shape) เป็น 600 * w * h * 3 โดยที่ w, h คือความกว้าง และความสูงของรูปภาพที่เหมาะสมกับ model ที่เลือกใช้ ( NASNetMobile w=224 h=224, ResNet50 w=224 h=224, InceptionV3 w=299 h=299) 
2) y_image : เก็บข้อมูล Class ของรูปภาพมีค่าเป็น 0, 1, 2
3) class_seq : เก็บข้อมูล dummy ของตัวเลขแต่ละ class 0-atmosphere, 1-food, 2-menu

# Training Strategy

# Result/Discussion/Conclusion

# Network Architecture
## NASNetMobile
NASNet ถูกสร้างขึ้นมาโดยมีวัตถุประสงค์ เพื่อให้ตัวโมเดลเข้ามาช่วยค้นหา Architecture CNN ที่เหมาะสมที่สุด โดยจะเรียกส่วนนี้ว่า Neural Architecture Search (NAS) ที่ถูกพัฒนาบนข้อมูล CIFAR10 และ ImageNet โดย Google Brain
ทั้งนี้ NASNetMobile เป็นเวอร์ชั่นย่อส่วนของ NASNet ตัวใหญ่ที่จะมีน้อยกว่า โดยที่ตัว NASNetMobile นี้มีจำนวน Layer ทั้งหมด 769 Layer และจำนวน Parameter ทั้งหมด 4,269,716 ตัว ซึ่งเราได้นำ pre training โมเดล ( NASNetMobile + Weight) มาเป็นส่วน Feature Extractor และเมื่อนำมารวมกับส่วน Classification Layer จะมีจำนวน Layer ทั้งหมด 775 Layer และจำนวน Parameter ทั้งหมด 17,780,119 ตัว 
<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/158823955-4444809f-61de-4756-8d8e-9314846f959a.png">
</p>

### Freeze All Layers 

### Unfreeze Layer 752 - 769
นำ NASNetMobile มา unfreeze Weight ใน Layer ตั้งแต่ 752 - 769 เพื่อให้โมเดลปรับจูนน้ำหนักให้เหมาะสมกับข้อมูลในระหว่างที่ Train ได้ ซึ่งเราหวังว่าจะได้ model ที่มีค่า accuracy/loss ที่ดีขึ้น ในขั้นตอนการ Train โมเดลเราดำเนินการบน Google-Colab (gpu : Tesla V100-SXM2-16GB)  โดยผลจากการ train ได้ผลลัพธ์ดีที่สุดมี loss อยู่ที่ 0.208, accuracy อยู่ที่ 0.950 ( atmosphere : 0.925, food : 0.953, menu : 0.973) และมี roc อยู่ที่ 0.994 (atmosphere : 0.994, food : 0.993, menu : 0.994)

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/87576892/158823694-cdfa9089-65ad-406d-a258-e12e632ba1ba.png">
  </br>(ภาพ accuracy และ loss ในระหว่างที่ train)

  <img width="800" src="https://user-images.githubusercontent.com/87576892/158823802-6f211d5c-6937-41a1-89e8-4eb7d6046dab.png">
  </br>(ภาพการทำ Grad-CAM เพื่อตรวจสอบโมเดล)
</p>

## ResNet50

### Freeze All Layers

### Unfreeze Layer: 171-174
จากการทำการทดลองการทำ Image Classifications เพื่อที่จะแยกรูปทั้งหมด 600 รูป (แบ่งออกเป็น รูปอาหาร 200 รูป, รูปเมนู 200 รูป และรูปบรรยากาศร้านอาหาร 200 รูป) ว่ารูปในกลุ่ม test set ที่นำมาทดสอบกับ model มีความแม่นยำมากน้อยเพียงใด โดยในส่วนของการทดลองนี้นั้น ทางเราจะใช้ ResNet50 พร้อม weight ที่ได้จากการ train ด้วย ImageNet มาใช้เฉพาะส่วน Feature Extractor โดยจะได้จำนวน layers ทั้งหมด 175 layers (0-174) และจำนวน Parameters ทั้งหมด 23,587,712 parameters โดยในการทดลองนี้นั้นเราจะทำการทดลองเพิ่มเติมจากเดิม (ResNet50: Freeze all layers) ตรงที่จะทำการ unfreeze weight ใน layer ตั้งแต่ 171 – 174 เพื่อให้ model ทำการ tuning weight ใหม่ให้เหมาะสมกับข้อมูลในระหว่างที่ทำการ train ได้ ซึ่งเมื่อนำมารวมกับส่วน classification layer ที่ทางเราได้ทำการกำหนดสร้างขึ้นมาใหม่เองนั้น จะทำให้มีจำนวน layers เพิ่มขึ้นเป็นทั้งหมด 181 layers และมีจำนวน parameters ทั้งหมด 49,541,763 parameters (เพิ่มขึ้น 110%) โดย tools ที่ทางทีมได้ทำการเลือกมาใช้ในการทดลองครั้งนี้นั้น คือ Google Colab (GPU: Tesla K80) ซึ่งผลลัพธ์ที่ได้จากการ train ที่ดีที่สุดมี loss อยู่ที่, accuracy อยู่ที่ 0.95 (atmosphere : 0.90, food : 0.95, menu : 0.97) และมี roc อยู่ที่ 0.99 (atmosphere : 0.99, food : 0.99, menu : 0.99)


<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/71161635/158633378-aa6a7b1c-fea3-4b76-8561-8d70a7f8d0b0.png">
 
  <img width="800" src="https://user-images.githubusercontent.com/71161635/158633403-9cfb142b-502d-4e00-87eb-9f82c2679a9b.png">
  </br>(ภาพ accuracy และ loss ในระหว่างที่ train)

  <img width="800" src="https://user-images.githubusercontent.com/71161635/158633696-127c8e6c-76a0-4f77-9854-ee7290ac551d.png">
  </br>(ภาพการทำ Grad-CAM เพื่อตรวจสอบโมเดล)
</p>


## Inception V3
<p align="center">
<img width="800" src="https://user-images.githubusercontent.com/87868128/158610993-a3ff03aa-f0bc-4a2d-bc5f-025fd487a3f3.png">
</p>
</p>
         จากภาพจะเห็นว่า InceptionV3 มีโครงสร้างเป็น 5 ส่วนหลักคือ Inception Module A ,Grid Size of Reduction step 1,Inception Module B ,Grid Size of Reduction Step2  Inception Module C ซึ่งความสามารถแยก output ได้ 1,000 classes
       Inception V3 เป็นโมเดลที่ได้รับการพัฒนาโดย Google ซึ่งได้รับรางวัลรองชนะเลิศสำหรับ Image Classification ใน ILSVR 2015 (78.1% accuracy ใน ImageNet dataset) มีโครงสร้าง Deep learning network ทั้งหมด 42 Layers มีจำนวน Parameter ทั้งหมด 21 ล้าน Parameter การพัฒนา Inception V3 มีจุดมุ่งหมายเพื่อให้ใช้ทรัพยากรในการคำนวณน้อยลง แต่ที่ได้ประสิทธิภาพที่สูงขึ้นโดยปรับปรุงจาก Inception architectures รุ่นก่อนหน้า ซึ่งแนวคิดในการออกแบบ Inception V3 สามารถแบ่งออกได้เป็น 4 ส่วนหลัก คือ
</br>1.	Factorization into Smaller Convolutions
เป็นการทำเพื่อช่วยลดการคำนวณเนื่องจากจะทำให้จำนวน Parameter น้อยลง ด้วยการแทนที่ convolutions อันใหญ่ด้วย convolutions อันที่เล็กกว่า เช่น แทนที่ 5 x 5 filter ด้วย 3 x 3 filter สองอันจะช่วยลดจำนวน Parameter ลงไปได้ 28%
</br>2.	Spatial Factorization into Asymmetric Convolutions
ทำ Asymmetric convolutions ต่อด้วยn x 1 filter เช่น แทนที่ 3 x 3 filter ด้วย 1 x 3 filter และ 3 x 1 filter ตามลำดับ จะทำให้ช่วยลดจำนวน Parameter ลงไปอีก 33%
</br>3.	Utility of Auxiliary classifiers
การใช้ Auxiliary classifiers เพื่อปรับปรุงการ convergence ใน deep neural network และส่วนใหญ่จะใช้เพื่อแก้ปัญหา Vanishing gradient ที่พบใน deep neural network ซึ่งใน inception V3 Auxiliary classifiers จะถูกใช้เป็น Regularizer
</br>4.	Efficient Grid Size Reduction
Grid size reduction จะถูกทำโดย max pooling และ average pooling และเพื่อแก้ปัญหาคอขวดในการคำนวณ Inception V3 เพิ่มวิธีการ ที่ทำให้ประสิทธิภาพในการทำ Grid size reduction เพิ่มมากขึ้น

</br>ประสิทธิภาพของ Inception V3
<p align="center">
<img width="800" src="https://user-images.githubusercontent.com/87868128/158611336-543fbd52-5255-4ffe-844c-d773232050ea.png">
</br>จากภาพเป็นผลจาก Multi-crop reported จะเห็นว่า Inception V3 มี error rate ที่น้อยมากเมื่อเทียบกับ Model ที่มีมาก่อนหน้า



### Freeze All Layers
ทางกลุ่มได้นำ InceptionV3 พร้อมน้ำหนักที่ได้จากการ train กับข้อมูล ImageNet มาเป็นส่วน Feature Extractor ซึ่งโมเดลจะมีจำนวน Layer ทั้งหมด 310 Layers และจำนวน Parameter ทั้งหมด 21,802,784 ตัว ทั้งนี้เราจะทำการ freezeทุกlayer เพื่อเป็น Base line ในกรณีที่ต้องการเปรียบเทียบประสิทธิภาพ unfreeze บางLayer ด้วยความคาดหวังว่า model ที่ unfreeze บาง layer จะให้ผลสรุปที่ดียิ่งขึ้น เมื่อนำ Feature Extractor มารวมกับ ส่วน Classification Layer จะมีจำนวน Layer ทั้งหมด 317  layer และจำนวน Parameter ทั้งหมด 55,621,155
การทดลองนี้ได้ train โมเดลผ่าน Google-Colab (gpu : Tesla-K80) โดยผลจากการ train ได้ผลลัพธ์ดีที่สุดมี loss อยู่ที่ 0.1798, accuracy อยู่ที่ 0.958 ( atmosphere : 0.925, food : 1.0, menu : 0.95) และมี roc อยู่ที่ 0.99 (atmosphere : 0.99, food : 1.0, menu : 1.0)

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/83268624/158615943-24e3f7bc-bca5-4bd5-845d-7582711b7611.png">
  <img width="800" src="https://user-images.githubusercontent.com/83268624/158616033-172fe746-9bbb-4a0e-aea5-f404840f67d7.png">
  </br>(ภาพ accuracy และ loss ในระหว่างที่ train)
  <img width="800" src="https://user-images.githubusercontent.com/83268624/158618409-bf8b2ade-93fd-4927-8a79-098b3c4c51b7.png">
  </br>(ภาพการทำ Grad-CAM เพื่อตรวจสอบโมเดล)
</p>

### Unfreeze Layer 299 – 310
ทางกลุ่มได้ทำการทดลองด้วยการนำ Model Inception V3 พร้อม weight ที่ได้จากการ train ด้วย ImageNet dataset มาใช้เฉพาะส่วน feature extractor ซึ่งจะได้จำนวน layers ทั้งหมด 310 layers และจำนวน parameter ทั้งหมด 21,802,784 parameter จากนั้นจึงทำการ unfreeze layer ให้สามารถ train ได้ตั้งแต่ layer ที่ 299 เป็นต้นไปเพื่อให้ model ปรับจูนค่า weight ใหม่ให้เหมาะสมกับข้อมูลที่ทางกลุ่มได้นำมาใช้ และเมื่อนำมารวมกับส่วน classification layer ที่ทางกลุ่มกำหนดขึ้นเองจะทำให้มีจำนวน layers เพิ่มขึ้นเป็นทั้งหมด 317 layers และมี parameter ที่สามารถ train ได้ทั้งหมด 34,213,123 parameter จากทั้งหมด 55,621,155 parameter โดยทำการ train model ผ่าน Google-Colab (gpu: Tesla P100-PCIE-16GB ) ซึ่งผลลัพธ์ที่ได้จากการ train ที่ดีที่สุดมี loss อยู่ที่ 0.3124, accuracy อยู่ที่ 0.9667 ( atmosphere : 0.90, food : 1.00, menu : 1.00) และมี roc อยู่ที่ 0.99 (atmosphere : 1.00, food : 0.99, menu : 1.00)
<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/87868128/158608598-bf8fd199-b4eb-4b7f-a4b9-3decbce942bf.png">
  <img width="800" src="https://user-images.githubusercontent.com/87868128/158608626-fd539324-33a4-45ba-8d01-07d2fc228085.png">
  </br>(ภาพ accuracy และ loss ในระหว่างที่ train)
  <img width="800" src="https://user-images.githubusercontent.com/87868128/158609146-59b3921e-fa82-40bb-905b-65f4a105850e.png">
  </br>(ภาพการทำ Grad-CAM เพื่อตรวจสอบโมเดล)
</p>

# Referance
- https://life.wongnai.com/internship-image-classification-wongnai-a1dbc2890766  
- https://iq.opengenus.org/inception-v3-model-architecture/
- https://blog.paperspace.com/popular-deep-learning-architectures-resnet-inceptionv3-squeezenet/
- https://arxiv.org/abs/1512.00567
- https://sh-tsang.medium.com/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c
- https://cloud.google.com/tpu/docs/inception-v3-advanced

# Member 
1) ณัฐภณ อัศวเหม 6310422052 (% contribution in this homework: 16.67%)
3) ดวงธิดา แซ่แต้ 6310422056 (% contribution in this homework: 16.67%)
4) เมธี ประเสริฐกิจพันธุ์ 6310422053 (% contribution in this homework: 16.67%)
5) พีรพัทธ ตั้งไพบูลย 6310422024 (% contribution in this homework: 16.67%)
6) วิชิต ชำนาญนาวา 6310422055 (% contribution in this homework: 16.67%)
7) ไตรทิพย์ ศุภศิริวัฒนา 6310422009 (% contribution in this homework: 16.67%)


