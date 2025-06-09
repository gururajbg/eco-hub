import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from 'next-themes';
import { AuthProvider } from './context/AuthContext';
import { DocumentProvider } from './context/DocumentContext';
import Navbar from './components/Navbar';
import Index from './pages/Index';
import EWasteManagement from './pages/EWasteManagement';
import BatteryRules from './pages/BatteryRules';
import AdminPage from './pages/AdminPage';
import ObjectDetection from './pages/ObjectDetection';
import Login from './pages/Login';
import NotFound from './pages/NotFound';
import Predictor from './pages/chem';
import CopperExtraction from './pages/CopperExtraction';

const App: React.FC = () => {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <Router>
        <AuthProvider>
          <DocumentProvider>
            <div className="min-h-screen flex flex-col">
              <Navbar />
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/e-waste" element={<EWasteManagement />} />
                <Route path="/battery-rules" element={<BatteryRules />} />
                <Route path="/admin" element={<AdminPage />} />
                <Route path="/object-detection" element={<ObjectDetection />} />
                <Route path="/login" element={<Login />} />
                <Route path="*" element={<NotFound />} />
                <Route path='/predict-data' element={<Predictor />} />
                <Route path='/copper-extraction' element={<CopperExtraction />} />
              </Routes>
            </div>
          </DocumentProvider>
        </AuthProvider>
      </Router>
    </ThemeProvider>
  );
};

export default App;
