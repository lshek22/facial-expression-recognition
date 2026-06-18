# facial-expression-recognition

Kaggle competition: [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)

---

## პროექტის აღწერა

ამ პროექტში ვმუშაობთ FER2013 dataset-ზე, რომელიც შეიცავს 35,887 grayscale სახის სურათს (48×48 პიქსელი), დაყოფილს 7 ემოციის კლასად: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

მიზანი იყო სხვადასხვა neural network არქიტექტურის გატესტვა, მათი შედარება და ყველა ექსპერიმენტის WandB-ზე დალოგვა.

---

## Dataset

| პარამეტრი | მნიშვნელობა |
|---|---|
| სულ სურათები | 35,887 |
| გარჩევადობა | 48×48 grayscale |
| კლასების რაოდენობა | 7 |
| Train set | ~25,838 (90%) |
| Validation set | ~2,871 (10%) |
| Test set | test.csv |

<img width="560" height="491" alt="image" src="https://github.com/user-attachments/assets/60f00b07-a50b-4406-82fe-80bcb88692d0" />


**Class imbalance პრობლემა:**
- Happy: 8,989 სურათი (ყველაზე მეტი)
- Disgust: 547 სურათი (ყველაზე ცოტა)

---

## რეპოზიტორიის სტრუქტურა

facial-expression-recognition/
    
    ├── notebooks/
    │   ├── experiment_simple_cnn.ipynb    ← Model 1
    │   ├── experiment_deep_cnn.ipynb      ← Model 2
    │   ├── experiment_resnet18.ipynb      ← Model 3
    │   └── inference.ipynb                ← საბოლოო პრედიქცია
    └── README.md

---

## გამოყენებული მიდგომები

### Preprocessing & Augmentation

ყველა მოდელისთვის გამოვიყენე:
- Normalization: პიქსელების გაყოფა 255-ზე → [0, 1] range
- Random Horizontal Flip
- Random Rotation (±10°)

---

## მოდელი 1 — Simple CNN (Baseline)

### არქიტექტურა

    Input (1, 48, 48)
    Conv2d(1, 32) → ReLU → MaxPool2d
    Conv2d(32, 64) → ReLU → MaxPool2d
    Flatten
    Linear(9216, 256) → ReLU
    Linear(256, 7)

### გადაწყვეტილების დასაბუთება
დავიწყე ყველაზე მარტივი არქიტექტურით — 2 convolutional layer — რათა დამეყენებინა baseline და გამეგო რა ხდება regularization-ის გარეშე.

### Forward & Backward Check
- Output shape: (64, 7)
- Initial loss: ~1.94 (მოსალოდნელი ln(7) = 1.9459)
- ყველა layer-ს აქვს gradient
- Single batch overfit test: 20 step-ში loss 1.94 → 0.3

### ჩატარებული ექსპერიმენტები

| Run | Optimizer | LR | Train Acc | Val Acc | დიაგნოზი |
|---|---|---|---|---|---|
| SimpleCNN-Adam-lr0.001 | Adam | 0.001 | 65.8% | 54.0% | Overfitting |
| SimpleCNN-Adam-lr0.01 | Adam | 0.01 | 25.1% | 25.7% | Collapsed |
| SimpleCNN-SGD-lr0.01 | SGD | 0.01 | 59.4% | 54.9% | good |

### ანალიზი

**Adam lr=0.001:** მოდელი კარგად სწავლობს epoch 12-მდე, შემდეგ train accuracy იზრდება მაგრამ val accuracy ჩერდება — **overfitting**. regularization არ გვაქვს, ამიტომ მოდელი train set-ს იზეპირებს.

**Adam lr=0.01:** LR ძალიან მაღალია — პირველივე epoch-ში loss explode-ს გაკეთდა, მოდელი ერთ კლასს წინასწარმეტყველებს (25% ≈ random for dominant class). **LR too high → model collapse**.

**SGD lr=0.01:** ყველაზე ჯანსაღი run — train და val accuracy ერთად იზრდება, overfitting ნაკლებია. SGD-ს momentum უფრო სტაბილური gradient updates-ი აქვს ამ შემთხვევაში.

<img width="1787" height="495" alt="image" src="https://github.com/user-attachments/assets/9785815e-0a81-454c-89e2-d7dad5820b48" />


---

## მოდელი 2 — Deep CNN with BatchNorm & Dropout

### არქიტექტურა

    Input (1, 48, 48)
    Conv2d(1, 32) → BatchNorm2d → ReLU → MaxPool2d → Dropout2d(0.25)
    Conv2d(32, 64) → BatchNorm2d → ReLU → MaxPool2d → Dropout2d(0.25)
    Conv2d(64, 128) → BatchNorm2d → ReLU → MaxPool2d → Dropout2d(0.25)
    Flatten
    Linear(4608, 512) → ReLU → Dropout(p)
    Linear(512, 256) → ReLU → Dropout(p)
    Linear(256, 7)
### გადაწყვეტილების დასაბუთება
Model 1-ში დავინახეთ overfitting. გადავწყვიტე დამეტოვებინა:
- **BatchNorm** — training სტაბილიზაციისთვის, faster convergence
- **Dropout** — overfitting-ის შესამცირებლად
- **მე-3 Conv block** — უფრო კომპლექსური features-ის სასწავლად
- **StepLR scheduler** — LR-ის თანდათანობით შემცირებისთვის

### ჩატარებული ექსპერიმენტები

| Run | Dropout | Weight Decay | Train Acc | Val Acc | დიაგნოზი |
|---|---|---|---|---|---|
| DeepCNN-dropout0.5 | 0.5 | 0 | 44.9% | 51.0% | Underfitting |
| DeepCNN-dropout0.3 | 0.3 | 0 | 50.3% | 53.8% | Best |
| DeepCNN-weightdecay | 0.5 | 1e-4 | 46.5% | 50.5% | Underfitting |

### ანალიზი

**dropout=0.5:** ძალიან ბევრი regularization + StepLR-ი ძალიან სწრაფად ამცირებს LR-ს (0.001 → 0.0001 → 0.00001 სულ 20 epoch-ში). შედეგად მოდელი ვერ სწავლობს საკმარისად — **underfitting**.

**dropout=0.3:** ნაკლები dropout საშუალებას აძლევს მოდელს უფრო მეტი ისწავლოს. საუკეთესო ბალანსი ამ სერიაში.

**weight decay:** dropout=0.5 + weight decay = ორმაგი regularization, რაც კიდევ უფრო აძლიერებს underfitting-ს.

**Note:** StepLR-ი step_size=7, gamma=0.1-ით ძალიან aggressive-ია 20 epoch-ისთვის. LR ძალიან სწრაფად კვდება convergence-მდე.


<img width="1790" height="495" alt="image" src="https://github.com/user-attachments/assets/6da1beb1-9b70-49ec-85c9-1cd5e08498e4" />

---

## მოდელი 3 — ResNet18 (Transfer Learning)

### არქიტექტურა
    Input (1, 48, 48)
    Conv2d(1, 64, kernel=7) [modified for grayscale]
    ResNet18 Backbone (pretrained on ImageNet)
    Dropout(0.5)
    Linear(512, 7)
### გადაწყვეტილების დასაბუთება
ResNet18 ავირჩიე რადგან:
- Residual connections vanishing gradient-ს წყვეტს ღრმა ქსელებში
- ImageNet-ზე pretrained weights გვაძლევს კარგ starting point-ს
- FER2013-ზე კარგად მუშაობს literature-ში

FER2013 grayscale-ია, ResNet კი RGB-ს ელის — პირველი Conv layer შევცვალე 1 channel-ზე.

### ჩატარებული ექსპერიმენტები

| Run | Pretrained | Backbone | Train Acc | Val Acc | დიაგნოზი |
|---|---|---|---|---|---|
| finetune-all | ✓ | Trainable | 72.9% | 56.3% | Overfitting |
| frozen-backbone | ✓ | Frozen | 26.8% | 28.8% | Severe Underfitting |
| **scratch** | ✗ | Trainable | **76.1%** | **61.6%** | Best |

### ანალიზი

**Fine-tune all:** pretrained weights კარგ starting point-ს იძლევა, მაგრამ train/val gap დიდია (72.9% vs 56.3%) — **overfitting**. მეტი regularization ან augmentation დაეხმარებოდა.

**Frozen backbone:** ყველაზე სუსტი შედეგი. მიზეზი: ImageNet features RGB high-resolution სურათებზეა trained. 48×48 grayscale სახის სურათებზე ეს features არ გადადის კარგად — backbone "არასწორ" features-ს ამოიცნობს.

**From scratch:** მოულოდნელად საუკეთესო შედეგი მიზეზი: ResNet-ის არქიტექტურა (residual connections, batch norm) თავისთავადძლიერია. Higher LR (0.001 vs 0.0001) + CosineAnnealing უფრო კარგად მუშაობს ამ dataset-ზე. მოდელი task-specific features-ს სწავლობს scratch-იდან.

---

## საერთო შედეგების შედარება

| მოდელი | Val Acc | მთავარი პრობლემა |
|---|---|---|
| SimpleCNN Adam lr=0.001 | 54.0% | Overfitting epoch 12-დან |
| SimpleCNN Adam lr=0.01 | 25.7% | LR collapse |
| SimpleCNN SGD lr=0.01 | 54.9% | საუკეთესო SimpleCNN |
| DeepCNN dropout=0.3 | 53.8% | StepLR ძალიან aggressive |
| DeepCNN dropout=0.5 | 51.0% | Underfitting |
| ResNet18 finetune | 56.3% | Overfitting |
| ResNet18 frozen | 28.8% | Features არ გადადის |
| **ResNet18 scratch** | **61.6%** | **საუკეთესო საერთო** |

---

## WandB Experiments

ყველა ექსპერიმენტი დალოგილია WandB-ზე:
[WandB Project](https://wandb.ai/lshek22-free-university-of-tbilisi-/face%20expression%20recognition)

---

