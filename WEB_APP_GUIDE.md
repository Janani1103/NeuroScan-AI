# NeuroScanAI Web Application Guide

## 🎨 Professional DarkMagenta Theme

The web application features a sophisticated medical-grade interface with a DarkMagenta color scheme:

- **Primary Color**: #8B008B (DarkMagenta)
- **Secondary Color**: #4B004B (Darker Magenta)
- **Accent Color**: #E6A8D7 (Light Pink)
- **Background**: #121212 (Dark Background)
- **Card Background**: #1E1E1E
- **Text**: #FFFFFF (White)
- **Border**: #383838

## 📱 Pages & Navigation

### 🏠 Dashboard
**Purpose**: Main landing page with overview and quick actions

**Features**:
- Welcome message and system overview
- Statistics cards (Training Images, Model Accuracy, Tumor Classes, Predictions Made)
- Quick upload section with drag-and-drop
- Recent prediction activity feed
- Model status indicator

**Navigation**: Sidebar → Dashboard

---

### 🖼️ Upload MRI
**Purpose**: Dedicated page for uploading MRI images

**Features**:
- Large drag-and-drop upload area
- File type validation (JPG, JPEG, PNG)
- Image preview after upload
- Image details display (filename, size, mode)
- Direct navigation to prediction page

**Navigation**: Sidebar → Upload MRI

---

### 🧠 Predict Tumor
**Purpose**: Display prediction results with detailed analysis

**Features**:
- Uploaded MRI image display
- **Prediction Result Card**:
  - Predicted tumor class (Glioma, Meningioma, Pituitary, No Tumor)
  - Confidence score (percentage)
  - Risk level indicator (High/Medium/Low)
- **Class Probabilities**: Progress bars for all 4 classes
- **Grad-CAM Visualization**:
  - Original MRI image
  - Heatmap overlay showing model attention
  - Side-by-side comparison
- **Action Buttons**:
  - Download Report (coming soon)
  - New Prediction
- **Medical Disclaimer**: Prominent warning about educational use

**Navigation**: Upload → Predict Button or Sidebar → Predict Tumor

---

### 📈 Analytics
**Purpose**: View performance metrics and prediction trends

**Features**:
- **Summary Statistics**:
  - Total predictions made
  - Average confidence score
  - Most common prediction class
- **Prediction Distribution Chart**:
  - Bar chart showing predictions by class
  - Color-coded by tumor type
  - Count labels on bars
- **Confidence Over Time**:
  - Line chart showing confidence trends
  - Prediction number on x-axis
  - Confidence percentage on y-axis
  - Grid lines for readability

**Navigation**: Sidebar → Analytics

---

### 🤖 Model Comparison
**Purpose**: Compare different deep learning architectures

**Features**:
- **Model Cards** for each architecture:
  - Custom CNN (92% accuracy)
  - **MobileNetV2 (96% accuracy)** ← Current model
  - ResNet50 (97% accuracy)
  - EfficientNetB0 (98% accuracy)
- **For each model**:
  - Accuracy percentage
  - Parameter count
  - Inference time
  - Description
  - Pros and cons lists
- **Visual indicators**:
  - "CURRENT" badge for active model
  - Green border for current model
  - Color-coded statistics

**Navigation**: Sidebar → Model Comparison

---

### 📄 Reports
**Purpose**: View and manage prediction history

**Features**:
- **Search**: Filter predictions by class name
- **Date Filter**: All time, Today, Last 7 days, Last 30 days
- **Report Cards** for each prediction:
  - Report number and timestamp
  - Predicted class and confidence
  - All class probabilities
  - View Details button
  - Download PDF button
- **Export Options**:
  - Export to CSV
  - Export to JSON
- **Empty State**: Message when no predictions exist

**Navigation**: Sidebar → Reports

---

## 🎯 User Workflow

### First-Time User
1. **Dashboard** → Welcome page with overview
2. **Upload MRI** → Upload first image
3. **Predict Tumor** → View prediction results
4. **Analytics** → View system performance
5. **Reports** → Access prediction history

### Returning User
1. **Dashboard** → Check recent activity
2. **Upload MRI** → Upload new image
3. **Predict Tumor** → Get instant results
4. **Reports** → Download previous reports

### Research/Analysis
1. **Analytics** → View performance trends
2. **Model Comparison** → Compare architectures
3. **Reports** → Export data for analysis

---

## 🎨 UI Components

### Stat Cards
Gradient background with DarkMagenta theme
- Large value display
- Descriptive label
- Used on Dashboard and Analytics

### Upload Box
Dashed border with hover effects
- Icon and instructions
- Drag-and-drop support
- Responsive design

### Result Box
Gradient background with centered content
- Prediction class (large)
- Confidence score (very large)
- Risk level badge

### Risk Level Badges
- **High**: Red background (#FF4444)
- **Medium**: Orange background (#FFBB33)
- **Low**: Green background (#00C851)

### Progress Bars
- Dark background track
- DarkMagenta fill
- Percentage label
- Used for class probabilities

### Charts
- Dark background (#1E1E1E)
- White text labels
- DarkMagenta primary colors
- Grid lines for readability

---

## 🔧 Technical Features

### Session State Management
- Page navigation state
- Uploaded image storage
- Prediction results storage
- Prediction history persistence

### Model Caching
- Model loaded once per session
- Class names cached
- Last conv layer name cached

### Responsive Design
- Works on desktop and tablet
- Sidebar navigation
- Flexible grid layouts
- Mobile-friendly cards

### Error Handling
- Model not found detection
- Image validation
- Graceful fallbacks
- User-friendly error messages

---

## 🚀 Launching the Application

```bash
streamlit run app.py
```

**Default URL**: http://localhost:8501

**Configuration**:
- Edit `config.py` for model paths
- Edit `THEME` dictionary in `app.py` for colors
- Modify page functions for custom features

---

## 📊 Data Flow

1. **Upload** → Image stored in session state
2. **Predict** → Model inference → Results stored
3. **Analytics** → Aggregates session history
4. **Reports** → Displays all predictions
5. **Export** → Downloads session data

---

## 🎯 Customization

### Add New Page
1. Create render function: `render_new_page()`
2. Add to sidebar menu in `render_sidebar()`
3. Add routing in `main()`
4. Update page state

### Modify Theme
Edit `THEME` dictionary in `app.py`:
```python
THEME = {
    'primary': '#8B008B',
    'secondary': '#4B004B',
    'accent': '#E6A8D7',
    # ... etc
}
```

### Add New Model
Edit `models_data` in `render_models_page()`:
```python
{
    'name': 'NewModel',
    'accuracy': '99%',
    'params': 'X.XM',
    'inference_time': 'Xms',
    'description': '...',
    'pros': [...],
    'cons': [...]
}
```

---

## 🏥 Medical Compliance Features

### Disclaimer
- Prominent medical warning
- Educational use only
- Consult professionals message
- Displayed on prediction page

### Risk Assessment
- Automatic risk level calculation
- Visual risk indicators
- Confidence-based classification
- Clear communication

### Data Privacy
- Session-based storage
- No persistent database
- Client-side processing
- Export capabilities

---

## 📈 Future Enhancements

### Planned Features
- [ ] PDF report generation
- [ ] User authentication
- [ ] Cloud model deployment
- [ ] Real database integration
- [ ] Batch image processing
- [ ] Model versioning
- [ ] A/B testing interface
- [ ] Advanced filtering in reports

### Potential Integrations
- [ ] DICOM file support
- [ ] PACS system integration
- [ ] Electronic health records
- [ ] Telemedicine platforms
- [ ] Hospital information systems

---

## 🆘 Troubleshooting

### Model Not Loading
- Check `models/best_model.keras` exists
- Verify TensorFlow version compatibility
- Check file permissions

### Session State Issues
- Clear browser cache
- Restart Streamlit
- Check for JavaScript errors

### Styling Problems
- Verify CSS injection
- Check browser compatibility
- Clear Streamlit cache

### Performance Issues
- Reduce image size in config
- Enable model caching
- Optimize data loading

---

**For technical support, refer to the main README.md or QUICKSTART.md**
