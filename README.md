🌾 Smart Agriculture: AI-Powered Plant Disease Detection

An AI-based system that detects plant diseases from leaf images using a custom-trained ResNet9 deep learning model.
This project aims to assist farmers and researchers by providing instant disease identification, descriptions, and remedies for various crops.

🚀 Features

🧠 Deep Learning Model (ResNet9) trained on 38 plant disease classes

🌿 Instant Image Prediction — upload a leaf photo and get:

Disease name

Confidence score

Disease description

Recommended remedies

💡 FastAPI Backend for API deployment

🖼️ Simple Web UI to upload and test images

☁️ Integrated with Hugging Face Spaces for cloud hosting

🧩 Supported Crops & Diseases

Supports 38 classes including:

Crop	Diseases Detected
🍎 Apple	Apple Scab, Black Rot, Cedar Apple Rust
🌽 Corn	Common Rust, Northern Leaf Blight, Gray Leaf Spot
🍇 Grape	Black Rot, Leaf Blight, Esca (Black Measles)
🍅 Tomato	Early Blight, Late Blight, Mosaic Virus, etc.
🥔 Potato	Early Blight, Late Blight
🍑 Peach	Bacterial Spot
🫑 Bell Pepper	Bacterial Spot
🍓 Strawberry	Leaf Scorch
🍊 Orange	Citrus Greening (HLB)
🌾 Soybean, Blueberry, Raspberry, Squash	Healthy/Unhealthy Classification
🧠 Model Information

Architecture: Custom ResNet9

Framework: PyTorch

Trained On: PlantVillage Dataset (38 classes)

File Format: .pth (PyTorch checkpoint)

Hosted On: Hugging Face Hub

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Kritimbist/smart-agriculture.git
cd smart-agriculture

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate   # for Windows
# OR
source venv/bin/activate   # for Linux/Mac

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the FastAPI Server
uvicorn main:app --reload

5️⃣ Open in Browser
http://127.0.0.1:8000

🧪 Example Output
Input Image	Prediction	Confidence	Description	Remedy

	Tomato___Early_blight	0.98	Early blight causes concentric brown rings on leaves.	Remove infected leaves and apply fungicides like chlorothalonil.
🌍 Deployment

This app can be deployed to:

Hugging Face Spaces (recommended for demo)

Render / Vercel / Railway (for full FastAPI hosting)

Local or Cloud Server (via Docker or manual deployment)

📦 Folder Structure
smart-agriculture/
│
├── main.py                # FastAPI server
├── model.py               # Model architecture & prediction logic
├── templates/
│   └── index.html         # Frontend upload UI
├── static/                # CSS/JS/image assets
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── model_weights.pth      # Trained model (downloaded from Hugging Face)

🧑‍💻 Tech Stack

Backend: FastAPI

Model: PyTorch

Frontend: HTML, CSS, Jinja2

Deployment: Hugging Face Spaces / GitHub Pages

Utilities: torchvision, Pillow, numpy

🌱 Future Improvements

✅ Add multilingual support (Nepali, English)

✅ Add fertilizer & pesticide recommendations

✅ Integrate live camera prediction

✅ Add mobile app support (Flutter / React Native)

🤝 Contributing

Contributions are welcome!
If you'd like to improve this project:

Fork the repo

Create a new branch (feature/your-feature)

Commit your changes

Push the branch and open a pull request

🧾 License

This project is licensed under the MIT License — feel free to use and modify it.

✨ Author

👨‍💻 Kritim Bista
📧 kritimbista7@gmail.com

🌐 GitHub Profile

🤖 Hugging Face Space
