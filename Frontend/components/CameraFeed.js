'use client';
import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';

const CameraFeed = ({ onAnalysisComplete }) => {
  const webcamRef = useRef(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [localStats, setLocalStats] = useState({ face_detected: false });

  const captureAndAnalyze = useCallback(async () => {
    if (isProcessing || !webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    setIsProcessing(true);

    try {
      // Convert Base64 to Blob
      const blob = await fetch(imageSrc).then(r => r.blob());
      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');

      // Send to your Hugging Face Backend
      // Ensure NEXT_PUBLIC_API_URL is set in .env.local (e.g., https://your-space.hf.space)
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/analyze-frame`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!data.error) {
        setLocalStats({ face_detected: data.face_detected });
        
        // PASS DATA UP TO PARENT PAGE
        if (onAnalysisComplete) {
          onAnalysisComplete(data);
        }
      }
    } catch (error) {
      console.error("Frame Analysis Error:", error);
    } finally {
      setIsProcessing(false);
    }
  }, [webcamRef, isProcessing, onAnalysisComplete]);

  // Run every 500ms
  useEffect(() => {
    const interval = setInterval(captureAndAnalyze, 500);
    return () => clearInterval(interval);
  }, [captureAndAnalyze]);

  return (
    <div className="relative rounded-2xl overflow-hidden shadow-inner bg-black">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode: "user" }}
        className="w-full h-full object-cover transform scale-x-[-1]" // Mirror effect
      />
      {!localStats.face_detected && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-red-500/80 text-white px-4 py-1 rounded-full text-sm font-bold animate-pulse">
          No Face Detected
        </div>
      )}
    </div>
  );
};

export default CameraFeed;