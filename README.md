# 🌿 Plant Disease Classification System

A modern, web-based application for detecting and classifying plant diseases using deep learning. Built with Flask and TensorFlow, featuring a beautiful, farmer-friendly interface with real-time camera integration.

---

## 📋 Table of Contents

- [Features](#-features)
- [Supported Plants & Diseases](#-supported-plants--diseases)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Model Architecture](#-model-architecture)
- [UI/UX Features](#-uiux-features)
- [API Endpoints](#-api-endpoints)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🔍 Disease Detection
- **38 Disease Classes**: Classifies diseases across 9 different plant types
- **High Accuracy**: CNN-based model trained on large dataset
- **Real-time Analysis**: Fast prediction on uploaded images
- **Health Report**: Professional results display with disease badges

### 📷 Image Input Methods
- **File Upload**: Upload images from your device
- **Direct Camera Capture**: Take photos directly using device camera
- **Camera Switching**: Toggle between front and back cameras
- **Live Preview**: Real-time camera feed with viewfinder overlay

### 💡 Treatment Recommendations
- **Fertilizer Recommendations**: Specific fertilizer advice for each disease
- **Treatment Tips**: Practical tips for disease management
- **Tabbed Interface**: Easy navigation between fertilizer and tips
- **Expert Guidance**: Detailed recommendations based on disease type

### 🎨 Modern UI/UX
- **Agricultural Theme**: Deep forest green (#2D5A27) and earthy gold (#D4A373) color scheme
- **Responsive Design**: Fully optimized for mobile and laptop
- **Card-Based Layout**: Clean, modern card design
- **Smooth Animations**: Fade-in effects and scanning animations
- **High Contrast**: Optimized for outdoor/sunlight viewing

### 📱 Mobile Features
- **Hamburger Menu**: Collapsible navigation on mobile
- **Large Touch Targets**: Minimum 50px buttons for easy tapping
- **Full-Width Layout**: 95% width on mobile devices
- **Camera Integration**: Native camera access on mobile devices

### 💻 Desktop Features
- **Two-Column Layout**: Side-by-side upload and camera options
- **Full Navigation**: Always-visible navigation bar
- **Optimized Spacing**: Professional desktop layout

---

## 🌾 Supported Plants & Diseases

The system can detect diseases in the following plants:

### 🫑 Bell Pepper
- Bacterial Spot
- Healthy

### 🌿 Cassava
- Bacterial Blight (CBB)
- Brown Streak Disease (CBSD)
- Green Mottle (CGM)
- Healthy
- Mosaic Disease (CMD)

### 🌽 Corn (Maize)
- Cercospora Leaf Spot (Gray Leaf Spot)
- Common Rust
- Healthy
- Northern Leaf Blight

### 🍇 Grape
- Black Rot
- Esca (Black Measles)
- Healthy
- Leaf Blight (Isariopsis Leaf Spot)

### 🥭 Mango
- Anthracnose Fungal Leaf Disease
- Healthy Leaf
- Rust Leaf Disease

### 🥔 Potato
- Early Blight
- Healthy
- Late Blight

### 🌾 Rice
- Brown Spot
- Healthy
- Hispa
- Leaf Blast

### 🌹 Rose
- Healthy Leaf
- Rust
- Sawfly Slug

### 🍅 Tomato
- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Mosaic Virus
- Septoria Leaf Spot
- Spider Mites (Two-Spotted Spider Mite)
- Target Spot
- Yellow Leaf Curl Virus

**Total: 38 Disease Classes**

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Plant_Disease_Classification
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Download and Place Model File

The model file (`Team3model.h5`) is too large for GitHub. Download it from Google Drive:

**📥 [Download Team3model.h5 from Google Drive](https://drive.google.com/file/d/19WPGOuIAJfxC7qqO6JYZ4JoeqGwOuvQW/view)**

After downloading:
1. Place the `Team3model.h5` file in the project root directory
2. Ensure the file is named exactly `Team3model.h5`
3. The application will automatically load this model on startup

**Note**: The model file is required for the application to function. Without it, the disease detection feature will not work.

---

## 💻 Usage

### Starting the Application

1. **Activate Virtual Environment** (if not already activated)
   ```bash
   .\venv\Scripts\Activate.ps1  # Windows
   # or
   source venv/bin/activate     # Linux/Mac
   ```

2. **Run the Flask Application**
   ```bash
   python app.py
   ```

3. **Access the Application**
   - Open your browser and navigate to: `http://localhost:5000`
   - The application will start on port 5000 by default

### Using the Application

1. **Login**
   - Enter any email and password (authentication is bypassed for demo)
   - Click "Login"

2. **Upload or Capture Image**
   - **Option 1**: Click "Upload Image" and select a file from your device
   - **Option 2**: Click "Open Camera" to capture a photo directly

3. **View Results**
   - Disease name with color-coded badge (red for infected, green for healthy)
   - Fertilizer recommendations tab
   - Treatment tips tab
   - Uploaded/captured image with scanning animation

4. **Analyze Another Image**
   - Click "Analyze Another Image" to start over

---

## 📁 Project Structure

```
Plant_Disease_Classification/
│
├── app.py                          # Main Flask application
├── requirements.txt                 # Python dependencies
├── Team3model.h5                    # Trained CNN model
├── README.md                        # Project documentation
│
├── static/
│   ├── style.css                    # Shared stylesheet
│   └── uploads/                     # Uploaded images directory
│
├── templates/
│   ├── login.html                   # Login page
│   ├── home.html                    # Home page
│   ├── disease-recognition.html     # Disease detection page
│   └── prediction.html              # Results page
│
└── venv/                            # Virtual environment (created after setup)
```

---

## 🛠 Technologies Used

### Backend
- **Python 3.11**: Programming language
- **Flask 3.0.0**: Web framework
- **TensorFlow 2.15.0**: Deep learning framework
- **Keras 2.15.0**: High-level neural networks API
- **NumPy 1.24.3**: Numerical computing
- **Pillow 10.1.0**: Image processing

### Frontend
- **HTML5**: Markup language
- **CSS3**: Styling with modern features
- **JavaScript**: Client-side interactivity
- **Inter Font**: Modern typography (Google Fonts)

### Machine Learning
- **Convolutional Neural Network (CNN)**: Model architecture
- **Image Preprocessing**: Resizing, normalization
- **Transfer Learning**: Pre-trained model optimization

---

## 🧠 Model Architecture

The CNN model consists of:
- **Input Layer**: 256x256x3 (RGB images)
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for dimensionality reduction
- **Dropout Layers**: Regularization to prevent overfitting
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: 38 classes (softmax activation)

### Model Details
- **Total Parameters**: ~59 million
- **Training Accuracy**: ~65%
- **Validation Accuracy**: ~81%
- **Test Accuracy**: ~65%

### Model File Download
The trained model file (`Team3model.h5`) is available for download:

**📥 [Download Team3model.h5](https://drive.google.com/file/d/19WPGOuIAJfxC7qqO6JYZ4JoeqGwOuvQW/view)**

**Important**: 
- The model file is not included in the repository due to its large size
- You must download it separately and place it in the project root directory
- The file must be named `Team3model.h5` for the application to work

---

## 🎨 UI/UX Features

### Design System
- **Primary Color**: Deep Forest Green (#2D5A27)
- **Accent Color**: Earthy Gold (#D4A373)
- **Background**: Clean Off-White (#F8F9FA)
- **Typography**: Inter font family
- **Border Radius**: 16px for modern look

### Key UI Components
1. **Sticky Navigation Bar**: Always accessible navigation
2. **Card-Based Layout**: Clean, organized content cards
3. **Health Report Style Results**: Professional disease report display
4. **Scanning Animation**: Visual feedback during image processing
5. **Tabbed Interface**: Easy navigation between fertilizer and tips
6. **Responsive Grid**: Adapts to screen size

### Animations
- **Fade-in**: Smooth page load animations
- **Scanning Line**: Green laser scan over uploaded images
- **Button Hover**: Scale and shadow effects
- **Smooth Transitions**: All interactive elements

---

## 🔌 API Endpoints

### Routes

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Redirects to login page |
| GET/POST | `/login` | Login page and authentication |
| GET | `/home` | Home page with features and supported plants |
| GET/POST | `/disease-recognition` | Disease detection page (upload/camera) |
| GET | `/logout` | Logout and redirect to login |

### Form Submission
- **Endpoint**: `/disease-recognition`
- **Method**: POST
- **Content-Type**: `multipart/form-data`
- **Field**: `file` (image file)
- **Response**: Renders prediction page with results

---

## 📊 Treatment Recommendations

Each disease includes:

### 🌾 Fertilizer Recommendations
- Specific fertilizer types (e.g., Potassium Sulfate, NPK 10-10-10)
- Application methods and timing
- Nutrient requirements

### 💡 Treatment Tips
- Cultural practices
- Chemical treatments
- Prevention strategies
- Environmental management

Examples:
- **Bacterial Diseases**: Copper-based bactericides, pathogen-free seeds
- **Fungal Diseases**: Fungicides, proper spacing, crop rotation
- **Viral Diseases**: Vector control, resistant varieties
- **Healthy Plants**: Maintenance recommendations

---

## 📱 Responsive Design

### Mobile (< 992px)
- Hamburger menu navigation
- Single-column stacked layout
- Full-width buttons (95% screen width)
- Large touch targets (50px minimum)
- Camera view optimized for mobile
- Vertical scrolling only

### Desktop (≥ 992px)
- Full navigation bar
- Two-column layout for upload/camera
- Standard button sizes
- Side-by-side content arrangement
- Optimized spacing and padding

---

## 🔒 Security Features

- **Session Management**: Flask sessions for user authentication
- **File Upload Security**: Secure filename handling
- **Input Validation**: Form validation on client and server
- **Error Handling**: Graceful error messages

---

## 🚧 Future Enhancements

Potential improvements:
- [ ] User authentication system
- [ ] Disease history tracking
- [ ] Export reports (PDF)
- [ ] Multi-language support (Telugu ready)
- [ ] Batch image processing
- [ ] Disease severity assessment
- [ ] Treatment cost calculator
- [ ] Integration with agricultural databases

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 👨‍💻 Development

### Running in Development Mode
The application runs with `debug=True` by default, enabling:
- Auto-reload on code changes
- Detailed error messages
- Development server

### Production Deployment
For production, set `debug=False` in `app.py`:
```python
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

---

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

---

> **🌾 Empowering farmers with AI-powered plant disease detection**  
> © 2025 Plant Disease Classification System – All rights reserved.
