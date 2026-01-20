'use client';

import React, { useState, useEffect } from 'react';
import { api } from '@/lib/api';
// CHANGE 1: Import the new CameraFeed component
import CameraFeed from '@/components/CameraFeed'; 
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function MonitorPage() {
  const [analytics, setAnalytics] = useState({
    current_emotion: 'Neutral',
    confidence: 0,
    stress_level: 'Low',
    emotion_distribution: {}
  });

  // CHANGE 2: This function receives instant updates from CameraFeed
  const handleLiveUpdate = (data) => {
    setAnalytics(prev => ({
      ...prev,
      current_emotion: data.emotion,
      confidence: data.confidence,
      stress_level: data.stress_level,
      // We don't update distribution here because analyze-frame doesn't return it
    }));
  };

  // CHANGE 3: Keep fetching distribution data separately (e.g., every 2 seconds)
  useEffect(() => {
    const fetchDistribution = async () => {
      try {
        const data = await api.getAnalytics();
        // Only update the distribution part to avoid overwriting the live frame data
        setAnalytics(prev => ({
          ...prev,
          emotion_distribution: data.emotion_distribution
        }));
      } catch (error) {
        console.error('Error fetching distribution:', error);
      }
    };

    fetchDistribution();
    // We can poll this slower (e.g. 2000ms) since graphs don't need 60fps updates
    const interval = setInterval(fetchDistribution, 2000); 

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        <Link href="/Prateek/dashboard" className="inline-flex items-center gap-2 mb-6 text-lime-600 font-bold hover:text-lime-700">
          <ArrowLeft size={20} />
          Back to Dashboard
        </Link>

        <h1 className="text-4xl font-black text-gray-900 mb-8">LIVE EMOTION MONITORING</h1>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Video Feed Section */}
          <div className="lg:col-span-2">
            <div className="bg-white p-6 rounded-3xl shadow-lg h-full">
              <div className="flex justify-between items-center mb-4">
                 <h2 className="text-2xl font-black">LIVE FEED</h2>
                 <span className="animate-pulse flex h-3 w-3 rounded-full bg-red-500"></span>
              </div>
              
              <div className="aspect-video bg-gray-900 rounded-2xl overflow-hidden">
                {/* CHANGE 4: The new CameraFeed component */}
                <CameraFeed onAnalysisComplete={handleLiveUpdate} />
              </div>
              
              <p className="text-gray-400 text-sm mt-4 text-center">
                * Processing frames securely via MoodQuest AI
              </p>
            </div>
          </div>

          {/* Analytics Panel */}
          <div className="space-y-6">
            
            {/* Current Emotion Card */}
            <div className="bg-white p-6 rounded-3xl shadow-lg transition-all hover:scale-105 duration-300">
              <h3 className="text-xl font-black mb-4 text-gray-400">CURRENT EMOTION</h3>
              <p className="text-5xl font-black text-lime-600 mb-2">
                {analytics.current_emotion}
              </p>
              <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                <div 
                  className="bg-lime-600 h-2.5 rounded-full transition-all duration-500" 
                  style={{ width: `${analytics.confidence}%` }}
                ></div>
              </div>
              <p className="text-right text-xs text-gray-500 mt-1">
                {analytics.confidence}% Confidence
              </p>
            </div>

            {/* Stress Level Card */}
            <div className="bg-white p-6 rounded-3xl shadow-lg transition-all hover:scale-105 duration-300">
              <h3 className="text-xl font-black mb-4 text-gray-400">STRESS LEVEL</h3>
              <p className={`text-4xl font-black ${
                analytics.stress_level === 'Low' ? 'text-green-500' :
                analytics.stress_level === 'Medium' ? 'text-yellow-500' :
                'text-red-500'
              }`}>
                {analytics.stress_level}
              </p>
              <p className="text-sm text-gray-500 mt-2">
                Based on recent emotion history
              </p>
            </div>

            {/* Distribution Card */}
            <div className="bg-white p-6 rounded-3xl shadow-lg">
              <h3 className="text-xl font-black mb-4 text-gray-400">SESSION STATS</h3>
              {analytics.emotion_distribution && (
                <div className="space-y-3 max-h-64 overflow-y-auto custom-scrollbar">
                  {Object.entries(analytics.emotion_distribution)
                    .sort(([,a], [,b]) => b - a) // Sort by count (highest first)
                    .map(([emotion, count]) => (
                    <div key={emotion} className="flex justify-between items-center p-2 hover:bg-gray-50 rounded-lg">
                      <span className="font-bold text-gray-700">{emotion}</span>
                      <span className="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm font-medium">
                        {count}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}