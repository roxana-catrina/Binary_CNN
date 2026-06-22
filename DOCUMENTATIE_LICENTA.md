# Documentație Tehnică — Sistem de Detecție și Clasificare a Tumorilor Cerebrale

## 1. Prezentare Generală

Sistemul reprezintă o aplicație medicală bazată pe inteligență artificială care analizează imagini RMN (Rezonanță Magnetică Nucleară) cerebrale pentru a:

1. **Detecta** prezența unei tumori cerebrale (clasificare binară: tumoră / fără tumoră)
2. **Clasifica** tipul tumorii (gliom, meningiom, tumoare pituitară)
3. **Segmenta** și **localiza** tumoarea pe imagine, calculând dimensiunile aproximative

---

## 2. Arhitectura Sistemului

Sistemul este compus din trei componente principale care comunică între ele prin protocoale HTTP REST:

```
┌─────────────────┐       HTTP        ┌──────────────────┐       HTTP        ┌─────────────────┐
│                 │  ──────────────►  │                  │  ──────────────►  │                 │
│  Frontend       │                   │  Backend Java    │                   │  Server AI      │
│  (Angular)      │  ◄──────────────  │  (Spring Boot)   │  ◄──────────────  │  (Flask/Python) │
│                 │       JSON        │                  │       JSON        │                 │
└─────────────────┘                   └──────────────────┘                   └─────────────────┘
     Port: 4200                            Port: 8080                            Port: 5000
```

### 2.1 Fluxul de Comunicare

1. **Utilizatorul** încarcă o imagine RMN prin interfața Angular
2. **Frontend-ul Angular** trimite imaginea ca `multipart/form-data` către backend-ul Java
3. **Backend-ul Java (Spring Boot)** retransmite imaginea către serverul AI Python
4. **Serverul AI (Flask)** procesează imaginea prin modelele de deep learning
5. Rezultatul (predicție + segmentare) parcurge drumul invers până la utilizator

### 2.2 Formate de Comunicare

| Segment | Protocol | Format Date | Content-Type |
|---------|----------|-------------|--------------|
| Angular → Java | HTTP POST | multipart/form-data | multipart/form-data |
| Java → Python | HTTP POST | multipart/form-data | multipart/form-data |
| Python → Java | HTTP Response | JSON | application/json |
| Java → Angular | HTTP Response | JSON | application/json |

---

## 3. Componenta AI — Server Flask (Python)

### 3.1 Tehnologii Utilizate

| Bibliotecă | Versiune | Rol |
|-----------|----------|-----|
| **PyTorch** | 2.12+ | Framework de deep learning pentru inferență |
| **torchvision** | 0.27+ | Modele pre-antrenate (ResNet18) și transformări de imagine |
| **Flask** | 3.1+ | Server web REST API |
| **Flask-CORS** | 6.0+ | Permite cereri cross-origin de la Angular |
| **OpenCV** | 4.13+ | Procesare de imagine pentru segmentare |
| **Pillow (PIL)** | 12+ | Manipulare imagini |
| **pydicom** | 3.0+ | Citire fișiere DICOM medicale |
| **NumPy** | 2.4+ | Operații numerice pe matrici |

### 3.2 Endpoint-uri API

#### `GET /health`
Verifică starea serverului și a modelelor încărcate.

**Răspuns:**
```json
{
  "status": "healthy",
  "binary_model_loaded": true,
  "hybrid_model_loaded": true,
  "device": "cpu"
}
```

#### `POST /api/predict`
Predicție simplă — detectează tumoarea și clasifică tipul.

**Request:** `multipart/form-data` cu câmpul `file` (imagine JPG/PNG/DICOM)

**Răspuns (tumoră detectată):**
```json
{
  "success": true,
  "prediction": "tumor",
  "has_tumor": true,
  "confidence": 0.97,
  "probabilities": {
    "no_tumor": 0.03,
    "tumor": 0.97
  },
  "tumor_type": "meningioma",
  "tumor_type_confidence": 0.85,
  "tumor_type_probabilities": {
    "glioma": 0.10,
    "meningioma": 0.85,
    "pituitary": 0.05
  }
}
```

#### `POST /api/predict-with-segmentation`
Predicție completă cu segmentare vizuală și dimensiuni tumorale.

**Request:** `multipart/form-data` cu câmpul `file`
**Query param opțional:** `threshold` (0.1 - 0.9, implicit 0.5)

**Răspuns (include segmentare):**
```json
{
  "success": true,
  "prediction": "tumor",
  "has_tumor": true,
  "confidence": 0.97,
  "tumor_type": "glioma",
  "tumor_type_confidence": 0.82,
  "segmentation": {
    "overlay_image_base64": "iVBORw0KGgo...",
    "contour_image_base64": "iVBORw0KGgo...",
    "dimensions": {
      "width_pixels": 145,
      "height_pixels": 160,
      "width_mm": 68.0,
      "height_mm": 75.0,
      "area_pixels": 15267,
      "area_mm2": 3354.7,
      "tumor_percentage": 5.82,
      "pixel_spacing_mm": 0.469
    },
    "bounding_box": { "x": 120, "y": 85, "width": 145, "height": 160 },
    "tumor_area_pixels": 15267,
    "tumor_percentage": 5.82,
    "method": "image_processing"
  }
}
```

#### `POST /api/batch-predict`
Predicție pe mai multe imagini simultan.

**Request:** `multipart/form-data` cu câmpul `files` (multiple fișiere)

---

## 4. Modele de Inteligență Artificială

### 4.1 Modelul Binar — CNN_TUMOR

**Scop:** Clasificare binară (tumoră / fără tumoră)

**Arhitectură:**
```
Input: 3 × 256 × 256 (imagine RGB redimensionată)
    │
    ▼
Conv2d(3→8, k=3) → ReLU → MaxPool(2×2)
    │
    ▼
Conv2d(8→16, k=3) → ReLU → MaxPool(2×2)
    │
    ▼
Conv2d(16→32, k=3) → ReLU → MaxPool(2×2)
    │
    ▼
Conv2d(32→64, k=3) → ReLU → MaxPool(2×2)
    │
    ▼
Flatten → FC(num_flatten → 100) → ReLU → Dropout(0.25)
    │
    ▼
FC(100 → 2) → LogSoftmax
    │
    ▼
Output: 2 clase [no_tumor, tumor]
```

**Parametri:**
- Input: 3 canale RGB, 256×256 pixeli
- Filtre inițiale: 8 (se dublează la fiecare strat)
- Dropout: 25%
- Funcție de activare finală: LogSoftmax (se aplică `exp()` pentru probabilități)

**Antrenare:**
- Split date: 80% train / 20% test
- Dataset: ~4087 imagini fără tumoră + imagini cu tumoră
- Deduplicare prin hash MD5

### 4.2 Modelul Hybrid — HybridTumorClassifier

**Scop:** Clasificare multiclasă a tipului de tumoră (3 clase: gliom, meningiom, pituitar)

**Arhitectură inovatoare:** Combină două ramuri de extragere de features:

```
                    Input: 3 × 224 × 224
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐
    │  Custom CNN      │   │  ResNet18        │
    │  (4 blocuri      │   │  (pre-antrenat   │
    │   convoluționale)│   │   pe ImageNet)   │
    │                  │   │                  │
    │  Output: 50176   │   │  Output: 512     │
    │  features        │   │  features        │
    └────────┬─────────┘   └────────┬─────────┘
             │                      │
             └──────────┬───────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Fuziune         │
              │  (Concatenare)   │
              │  50176 + 512     │
              │  = 50688 features│
              └────────┬─────────┘
                       │
                       ▼
              FC(50688 → 512) → BN → ReLU → Dropout(0.5)
                       │
                       ▼
              FC(512 → 256) → BN → ReLU → Dropout(0.4)
                       │
                       ▼
              FC(256 → 3) → Softmax
                       │
                       ▼
              Output: [glioma, meningioma, pituitary]
```

**Ramura Custom CNN (4 blocuri):**
- Bloc 1: Conv2d(3→32) × 2 + BN + ReLU + MaxPool + Dropout2d(0.2)
- Bloc 2: Conv2d(32→64) × 2 + BN + ReLU + MaxPool + Dropout2d(0.2)
- Bloc 3: Conv2d(64→128) × 2 + BN + ReLU + MaxPool + Dropout2d(0.3)
- Bloc 4: Conv2d(128→256) × 2 + BN + ReLU + MaxPool + Dropout2d(0.3)

**Ramura ResNet18:**
- Model pre-antrenat pe ImageNet (1.2M imagini, 1000 clase)
- Se elimină ultimul layer FC, se păstrează doar extragerea de features
- Output: vector de 512 features

**Strategii de fuziune disponibile:**
1. **Concatenare** (utilizată): Concatenează vectorii de features → clasificator comun
2. **Adunare**: Proiectează ambele ramuri la aceeași dimensiune și le adună
3. **Atenție**: Mecanism de atenție care ponderează contribuția fiecărei ramuri

**Parametri de antrenare:**
- Input: 224×224 pixeli
- Batch size: 16
- Learning rate: 0.001
- Epoci: 20
- Normalizare: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 4.3 Preprocesarea Imaginilor

**Pentru modelul binar:**
```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Pentru modelul hybrid:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 5. Modulul de Segmentare a Tumorii

### 5.1 Abordare

Segmentarea nu folosește un model de deep learning dedicat (precum U-Net), ci o abordare bazată pe **procesare de imagine** care exploatează proprietățile vizuale ale tumorilor pe RMN:

- Tumorile sunt de obicei **mai luminoase** pe RMN cu contrast (T1 + gadolinium)
- Tumorile se află în **interiorul** parenchimului cerebral, nu la periferie
- Tumorile au formă relativ **compactă** (rotundă/ovală)

### 5.2 Pipeline de Segmentare

```
Imagine RMN originală
        │
        ▼
┌─────────────────────────────────┐
│ 1. Creare mască parenchim       │
│    - Threshold pentru foreground │
│    - Erodare 4% pentru excludere│
│      craniu și scalp            │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ 2. Detecție regiuni luminoase   │
│    - Percentila 75 ca threshold │
│    - Doar în interiorul         │
│      parenchimului              │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ 3. Curățare morfologică         │
│    - Close (umple goluri)       │
│    - Open (elimină zgomot)      │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ 4. Scoring componente conexe    │
│    - Compactitate (25%)         │
│    - Intensitate medie (25%)    │
│    - Distanță de la margine(30%)│
│    - Dimensiune rezonabilă (20%)│
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│ 5. Generare overlay vizual      │
│    - Contur roșu pe tumoră      │
│    - Bounding box verde         │
│    - Dimensiuni în pixeli și mm │
│    - Heatmap de intensitate     │
└─────────────────────────────────┘
```

### 5.3 Excluderea Craniului

Un pas critic este **excluderea craniului și scalpului** din analiză. Fără acest pas, marginile osoase (foarte luminoase pe RMN) ar fi detectate greșit ca tumoră.

```python
erosion_size = max(7, int(min(h, w) * 0.04))  # ~4% din dimensiunea imaginii
brain_mask = cv2.erode(outer_mask, kernel, iterations=1)
```

### 5.4 Scoring Multi-Criteriu

Fiecare regiune candidat primește un scor bazat pe:

| Criteriu | Pondere | Justificare |
|----------|---------|-------------|
| Distanță de la margine | 30% | Tumorile sunt interioare, nu la periferie |
| Compactitate | 25% | Tumorile au formă rotundă/ovală |
| Intensitate medie | 25% | Tumorile sunt luminoase pe RMN cu contrast |
| Dimensiune | 20% | Tumorile ocupă 3-45% din parenchim |

### 5.5 Calculul Dimensiunilor

Dimensiunile sunt estimate pe baza unui FOV (Field of View) standard de 240mm:

```python
pixel_spacing = 240.0 / image_width  # mm per pixel
tumor_width_mm = bounding_box_width * pixel_spacing
tumor_area_mm2 = tumor_area_pixels * pixel_spacing_x * pixel_spacing_y
```

**Notă:** Pentru dimensiuni exacte, sunt necesare metadatele DICOM (câmpul `PixelSpacing`).

---

## 6. Procesarea Fișierelor DICOM

### 6.1 Ce este DICOM?

DICOM (Digital Imaging and Communications in Medicine) este standardul internațional pentru imagistica medicală. Fișierele `.dcm` conțin:
- Datele pixel ale imaginii
- Metadate despre pacient, echipament, parametri de achiziție
- Informații despre spațiere pixeli (PixelSpacing)

### 6.2 Procesare

```python
class DicomProcessor:
    def read_dicom_file(file_path):
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array
        # Normalizare la 0-255
        pixel_array = (pixel_array - min) / (max - min) * 255
        return pixel_array
```

Serverul detectează automat fișierele DICOM (prin extensie `.dcm`/`.dicom` sau prin semnătura `DICM` la byte 128) și le convertește în imagini RGB pentru procesare.

---

## 7. Componenta Backend — Java Spring Boot

### 7.1 Rol

Backend-ul Java servește ca **intermediar** (proxy) între frontend-ul Angular și serverul AI Python:

- Gestionează autentificarea și autorizarea utilizatorilor
- Retransmite cererile de predicție către Flask
- Stochează istoricul predicțiilor în baza de date
- Gestionează logica de business

### 7.2 Comunicare cu Flask

```java
@PostMapping("/predict-with-segmentation")
public ResponseEntity<PredictionResponse> predictWithSegmentation(
        @RequestParam("file") MultipartFile file,
        @RequestParam(value = "threshold", defaultValue = "0.4") double threshold) {

    String pythonUrl = "http://localhost:5000/api/predict-with-segmentation?threshold=" + threshold;

    MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
    body.add("file", new ByteArrayResource(file.getBytes()) {
        @Override
        public String getFilename() {
            return file.getOriginalFilename();
        }
    });

    HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
    return restTemplate.postForEntity(pythonUrl, request, PredictionResponse.class);
}
```

**Important:** Se folosește `ByteArrayResource` (nu `MultipartFile` direct) pentru compatibilitate cu Flask.

---

## 8. Componenta Frontend — Angular

### 8.1 Rol

Interfața web permite utilizatorului să:
- Încarce imagini RMN (JPG, PNG, DICOM)
- Vizualizeze rezultatul predicției (tumoră/fără tumoră, tip)
- Vizualizeze segmentarea (contur roșu pe tumoră, heatmap)
- Vadă dimensiunile aproximative ale tumorii

### 8.2 Afișarea Segmentării

Imaginile segmentate sunt primite ca string-uri Base64 și afișate direct:

```html
<img [src]="'data:image/png;base64,' + segmentation.contour_image_base64"
     alt="Tumor contour" />
```

### 8.3 Model de Date TypeScript

```typescript
interface PredictionResponse {
  success: boolean;
  prediction: string;        // "tumor" | "no_tumor"
  has_tumor: boolean;
  confidence: number;
  tumor_type?: string;       // "glioma" | "meningioma" | "pituitary"
  tumor_type_confidence?: number;
  segmentation?: {
    overlay_image_base64: string;
    contour_image_base64: string;
    dimensions: TumorDimensions;
    bounding_box: BoundingBox;
    tumor_percentage: number;
  };
}
```

---

## 9. Structura Datelor de Antrenare

```
data/
├── binary/                          # Date pentru clasificare binară
│   ├── tumor/                       # Imagini cu tumoră (raw)
│   ├── no_tumor/                    # Imagini fără tumoră (~4087 fișiere)
│   ├── train/
│   │   ├── tumor/                   # 80% din imagini
│   │   └── no_tumor/
│   └── test/
│       ├── tumor/                   # 20% din imagini
│       └── no_tumor/
│
└── multiclass/                      # Date pentru clasificare multiclasă
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        └── pituitary/
```

**Split-ul datelor** se face automat prin `config_binary/config.py`:
- 80% antrenare, 20% testare
- Randomizare cu `random.shuffle()`
- Deduplicare prin hash MD5 (evită imagini duplicate între train și test)

---

## 10. Fluxul Complet de Inferență

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUX DE INFERENȚĂ                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Imagine primită (JPG/PNG/DICOM)                             │
│         │                                                        │
│         ▼                                                        │
│  2. Conversie la RGB + Resize 256×256                           │
│         │                                                        │
│         ▼                                                        │
│  3. MODEL BINAR (CNN_TUMOR)                                     │
│     ┌─────────────────────────────┐                             │
│     │ Input: 3×256×256            │                             │
│     │ Output: [P(no_tumor), P(tumor)] │                         │
│     └─────────────┬───────────────┘                             │
│                   │                                              │
│         ┌─────────┴─────────┐                                   │
│         │                   │                                    │
│    no_tumor              tumor                                   │
│    (STOP)                  │                                     │
│                            ▼                                     │
│  4. Resize 224×224 + MODEL HYBRID                               │
│     ┌─────────────────────────────┐                             │
│     │ Custom CNN + ResNet18       │                             │
│     │ Fuziune: Concatenare        │                             │
│     │ Output: [P(glioma),         │                             │
│     │          P(meningioma),     │                             │
│     │          P(pituitary)]      │                             │
│     └─────────────┬───────────────┘                             │
│                   │                                              │
│                   ▼                                              │
│  5. SEGMENTARE (Image Processing)                               │
│     ┌─────────────────────────────┐                             │
│     │ Mască parenchim (fără craniu)│                            │
│     │ Detecție regiuni luminoase   │                            │
│     │ Scoring multi-criteriu       │                            │
│     │ Generare overlay + dimensiuni│                            │
│     └─────────────┬───────────────┘                             │
│                   │                                              │
│                   ▼                                              │
│  6. RĂSPUNS JSON                                                │
│     - Predicție binară + confidență                             │
│     - Tip tumoră + confidență                                   │
│     - Imagini segmentate (Base64)                               │
│     - Dimensiuni tumoră (px + mm)                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Fișiere Model Salvate

| Fișier | Conținut | Dimensiune Input |
|--------|----------|-----------------|
| `Brain_Tumor_model.pt` | Model binar complet (`torch.save(model)`) | 256×256 |
| `best_hybrid_model_hybrid_concat.pth` | State dict model hybrid (concat) | 224×224 |
| `best_model_multiclass.pth` | State dict model multiclasă | 224×224 |

---

## 12. Configurare și Rulare

### 12.1 Dependențe Python

```bash
pip install torch torchvision flask flask-cors opencv-python pydicom pillow numpy torchsummary
```

### 12.2 Pornire Server AI

```bash
python api_server.py
# Server pornește pe http://localhost:5000
```

### 12.3 Verificare Funcționare

```bash
# Health check
curl http://localhost:5000/health

# Predicție
curl -X POST -F "file=@brain_scan.jpg" http://localhost:5000/api/predict

# Predicție cu segmentare
curl -X POST -F "file=@brain_scan.jpg" http://localhost:5000/api/predict-with-segmentation
```

---

## 13. Considerații Tehnice

### 13.1 Compatibilitate CORS

Serverul Flask are CORS activat pentru a permite cereri de la frontend-ul Angular care rulează pe un alt port:

```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
```

### 13.2 Compatibilitate Spring Boot

Codul include gestionarea specifică pentru probleme de compatibilitate cu Spring Boot:
- Reset-ul poziției stream-ului (`file.stream.seek(0)`)
- Detectarea fișierelor corupte (bytes zero din multipart upload)
- Suport pentru `ByteArrayResource`

### 13.3 Suport GPU

Sistemul detectează automat disponibilitatea GPU NVIDIA (CUDA):

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Dacă GPU nu este disponibil, inferența se face pe CPU (mai lentă, dar funcțională).

### 13.4 Limitări Cunoscute

1. **Dimensiunile tumorii sunt estimative** — bazate pe un FOV standard de 240mm. Pentru dimensiuni exacte, sunt necesare metadatele DICOM.
2. **Segmentarea nu este un model dedicat** — folosește procesare de imagine, nu un model U-Net antrenat pe segmentare. Acuratețea depinde de contrastul imaginii.
3. **Clasificarea multiclasă** — modelul clasifică doar 3 tipuri de tumori (gliom, meningiom, pituitar). Alte tipuri nu sunt acoperite.

---

## 14. Diagrama de Clase — Modele AI

```
┌─────────────────────────────┐
│       CNN_TUMOR              │
│  (Clasificare Binară)        │
├─────────────────────────────┤
│ - conv1: Conv2d(3→8)        │
│ - conv2: Conv2d(8→16)       │
│ - conv3: Conv2d(16→32)      │
│ - conv4: Conv2d(32→64)      │
│ - fc1: Linear(flatten→100)  │
│ - fc2: Linear(100→2)        │
├─────────────────────────────┤
│ + forward(X) → LogSoftmax   │
└─────────────────────────────┘

┌─────────────────────────────┐
│  HybridTumorClassifier       │
│  (Clasificare Multiclasă)    │
├─────────────────────────────┤
│ - custom_features: Sequential│
│   (4 blocuri conv)           │
│ - resnet: ResNet18           │
│   (pre-antrenat ImageNet)    │
│ - fusion_classifier:         │
│   Sequential(FC layers)      │
├─────────────────────────────┤
│ + forward(X) → 3 clase      │
│ + fusion_type: 'concat'     │
└─────────────────────────────┘

┌─────────────────────────────┐
│    TumorClassifier           │
│  (Clasificare Multiclasă    │
│   - model alternativ)        │
├─────────────────────────────┤
│ - features: Sequential       │
│   (4 blocuri conv, mai adânc)│
│ - classifier: Sequential     │
│   (4 FC layers)              │
├─────────────────────────────┤
│ + forward(X) → 3 clase      │
└─────────────────────────────┘
```

---

## 15. Securitate și Validare

- **Validare fișiere:** Se verifică semnătura magic bytes (JPEG: FF D8 FF, PNG: 89 50 4E 47, DICOM: DICM la byte 128)
- **Protecție fișiere corupte:** Detectare fișiere cu zero bytes
- **Cleanup:** Fișierele temporare sunt șterse automat după procesare
- **Logging:** Toate operațiile sunt loggate pentru debugging

---

## 16. Rezumat Tehnologii

| Componentă | Tehnologie | Limbaj |
|-----------|-----------|--------|
| Server AI | Flask + PyTorch | Python 3.13 |
| Backend | Spring Boot | Java |
| Frontend | Angular | TypeScript |
| Modele DL | PyTorch (CNN + ResNet18) | Python |
| Segmentare | OpenCV | Python |
| Imagistică medicală | pydicom | Python |
| Comunicare | REST API (JSON) | HTTP |
