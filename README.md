# ⏱️ Control Work Time — Staff Scheduling & Timesheet OCR Platform

[![Live Demo](https://img.shields.io/badge/Live_Demo-Firebase_Hosting-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://control-work-time.web.app)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-Build-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)
[![OCR](https://img.shields.io/badge/OCR-Tesseract.js_%26_PDF.js-000000?style=for-the-badge)](https://tesseract.projectnaptha.com/)

**Control Work Time** is a client-side web application designed for hospitality and restaurant operations in France. It automates weekly and monthly shift scheduling under French labor rules (`39h`, `35h`, and `18h` contracts) and digitizes handwritten or PDF timesheets (*feuilles de pointage*) directly in the browser using local OCR.

🔗 **Live Application:** [https://control-work-time.web.app](https://control-work-time.web.app)

---

## ✨ Key Features

* **Automated Shift Simulation (*Simulation du planning*):** Generates balanced weekly and monthly schedules respecting contract targets (`39h00`, `35h00`, `18h00`), lunch/dinner split shifts, mandatory rest days, and Sunday afternoon closures.
* **Client-Side Timesheet OCR:** Extracts hours from photos and PDF scans (*Importer feuille de pointage*) using **Tesseract.js** (`fra.traineddata`) and **PDF.js** with an interactive validation modal before saving.
* **Midnight Split Calculation (*Ventilation minuit*):** Accurately splits overnight shifts across date boundaries without double-counting hours across reporting periods.
* **Zero-Cost Local-First Storage:** Persists all schedules, employee profiles, and service templates in the browser via `localStorage` & `IndexedDB`, with JSON backup/restore (*Données locales*) and PDF/print export.
* **Bilingual / French Hospitality UI:** Built specifically for French restaurant workflows with Light/Dark mode (`Clair / Sombre`).

---

## 🛠️ Tech Stack

* **Frontend:** React, TypeScript, Vite, Tailwind CSS
* **OCR & Document Processing:** Tesseract.js (French language model), PDF.js
* **Persistence:** Browser `IndexedDB` & `localStorage` (Zero-Backend Architecture)
* **Hosting & Deployment:** Firebase Hosting

---

## 🚀 Local Setup & Deployment

### Prerequisites
* Node.js (v18+) and `npm`
* Firebase CLI (`npm install -g firebase-tools`)

### Installation
```bash
git clone [https://github.com/michelspead-ai/Control-Work-Time.git](https://github.com/michelspead-ai/Control-Work-Time.git)
cd Control-Work-Time
npm install --legacy-peer-deps
npm run dev
