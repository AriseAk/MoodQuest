"use client";

import React, { useState, useEffect, useRef } from "react";
import Spline from "@splinetool/react-spline";
import Navbar from "@/components/Navbar";

export default function PetPage() {
  const [input, setInput] = useState("");
  const [reply, setReply] = useState("Hello! Ask me anything.");
  const [isLoading, setIsLoading] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  
  // We don't need a ref for the text box anymore, 
  // but we keep speechRef to manage audio memory
  const speechRef = useRef(null);

  // --- CLEANUP AUDIO ON LOAD ---
  useEffect(() => {
    if (typeof window !== "undefined") {
      window.speechSynthesis.cancel();
    }
    return () => {
      if (typeof window !== "undefined") {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const speakReply = (text) => {
    if (isMuted) return;

    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      speechRef.current = utterance;
      window.speechSynthesis.speak(utterance);
    }
  };

  const toggleMute = () => {
    const shouldBeMuted = !isMuted;
    setIsMuted(shouldBeMuted);
    if (shouldBeMuted) {
      window.speechSynthesis.pause();
    } else {
      window.speechSynthesis.resume();
    }
  };

  const handleReplay = () => {
    if (!reply || isLoading) return;
    speakReply(reply);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    setIsLoading(true);
    setReply("Thinking..."); // Bubble updates immediately

    try {
      const response = await fetch("http://127.0.0.1:5000/api/grok", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input.trim() }),
      });

      const data = await response.json();
      setReply(data.reply);
      speakReply(data.reply);

    } catch (error) {
      console.error("Error:", error);
      setReply("Sorry, I couldn't connect to my brain.");
    }

    setIsLoading(false);
    setInput("");
  };

  return (
  <div className="flex flex-col w-full h-screen bg-white">

    <Navbar/>

    <div className="flex w-full h-screen p-6 gap-6 bg-gray-50">
      
      {/* --- LEFT: SPLINE 3D SCENE + SPEECH BUBBLE --- */}
      <div className="w-2/3 relative h-full border rounded-2xl shadow-xl overflow-hidden bg-white">
        
        {/* The 3D Model */}
        <Spline scene="https://prod.spline.design/g7UNYT7AVq4LJ3eO/scene.splinecode" />

        {/* --- COMIC STYLE SPEECH BUBBLE --- */}
        {/* We position this absolutely inside the Spline container */}
        <div className="absolute top-10 left-10 right-10 z-20 flex flex-col items-center">
          <div className="relative max-w-2xl bg-white text-gray-800 text-lg font-medium px-8 py-6 rounded-3xl shadow-2xl border-2 border-gray-100 transition-all duration-300">
            
            {/* The Text Content */}
            {isLoading ? (
              <div className="flex items-center gap-2 text-gray-400">
                <span className="animate-bounce">●</span>
                <span className="animate-bounce delay-100">●</span>
                <span className="animate-bounce delay-200">●</span>
              </div>
            ) : (
              reply
            )}

            {/* The "Tail" of the bubble pointing down */}
            <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 w-6 h-6 bg-white border-b-2 border-r-2 border-gray-100 rotate-45"></div>
          </div>
        </div>

        {/* Mute Button (Floating on the 3D scene) */}
        <button
          onClick={toggleMute}
          className={`absolute top-6 right-6 z-30 p-3 rounded-full shadow-lg transition-all ${
            isMuted ? "bg-red-500 text-white" : "bg-white text-gray-700 hover:bg-gray-100"
          }`}
        >
          {isMuted ? (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 5L6 9H2v6h4l5 4V5z"></path><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
          )}
        </button>
      </div>

      {/* --- RIGHT: CONTROLS & INPUT --- */}
      <div className="w-1/3 flex flex-col gap-4">
        
        {/* Top Control Panel */}
        <div className="bg-white p-6 rounded-2xl shadow-lg flex flex-col gap-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-bold text-gray-800">Controls</h2>
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
          </div>
          
          <button
            onClick={handleReplay}
            disabled={isLoading || isMuted}
            className="w-full flex items-center justify-center gap-2 bg-blue-50 text-blue-600 py-3 rounded-xl font-semibold hover:bg-blue-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.3"/></svg>
            Replay Voice
          </button>
        </div>

        {/* Input Area (Takes remaining height) */}
        <div className="flex-1 bg-white p-6 rounded-2xl shadow-lg flex flex-col justify-between">
          <div>
            <h3 className="text-gray-500 font-medium mb-2">Your Message</h3>
            <p className="text-sm text-gray-400 mb-6">Type below to talk to your companion.</p>
          </div>
          
          <form onSubmit={handleSubmit} className="flex flex-col gap-3">
            <textarea
              placeholder="What's on your mind?"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
              className="w-full h-32 border border-gray-200 rounded-xl p-4 text-lg focus:outline-none focus:ring-2 focus:ring-lime-500 resize-none bg-gray-50"
              onKeyDown={(e) => {
                if(e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-lime-600 text-white py-4 rounded-xl font-bold text-lg hover:bg-lime-700 transition-all shadow-md active:scale-95 disabled:opacity-50"
            >
              {isLoading ? "Thinking..." : "Send Message"}
            </button>
          </form>
        </div>

      </div>
    </div>
    </div>
);
}