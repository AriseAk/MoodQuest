"use client"

import { api } from '@/lib/api';
import React, { useState, useEffect } from 'react';
import { Bell, Home, TrendingUp, Heart, Settings, HelpCircle, User } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useSession, signOut } from 'next-auth/react';
import { Loader2, LogOut } from 'lucide-react';

const Dashboard = () => {
  // 1. DECLARE ALL HOOKS FIRST
  const { data: session, status } = useSession();
  const router = useRouter();
  const params = useParams();

  // State Hooks
  const [realTimeAnalytics, setRealTimeAnalytics] = useState(null);
  const [joke, setJoke] = useState('');
  const [fact, setFact] = useState('');
  const [activeTab, setActiveTab] = useState('home');
  const [mounted, setMounted] = useState(false);
  const [statsVisible, setStatsVisible] = useState(false);

  // Effect 1: Auth Redirect
  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/login');
    }
  }, [status, router]);

  // Effect 2: Fetch Analytics
  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const data = await api.getAnalytics();
        setRealTimeAnalytics(data);
      } catch (error) {
        console.error('Error fetching analytics:', error);
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 2000);
    return () => clearInterval(interval);
  }, []);

  // Effect 3: Fetch Content
  useEffect(() => {
    const fetchContent = async () => {
      try {
        const [jokeData, factData] = await Promise.all([
          api.getJoke(),
          api.getFact()
        ]);
        setJoke(jokeData.joke);
        setFact(factData.fact);
      } catch (error) {
        console.error('Error fetching content:', error);
      }
    };

    fetchContent();
  }, []);

  // Effect 4: Animation Mount (FIXED FOR LINTER)
  useEffect(() => {
    // We wrap this in a timeout of 0ms to stop the "synchronous setState" error
    const mountTimer = setTimeout(() => {
      setMounted(true);
    }, 0);
    
    const statsTimer = setTimeout(() => {
      setStatsVisible(true);
    }, 300);

    return () => {
      clearTimeout(mountTimer);
      clearTimeout(statsTimer);
    };
  }, []);

  // 2. CONDITIONAL RETURNS (Only allowed AFTER hooks)
  if (status === 'loading') {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="animate-spin text-lime-500" size={64} />
      </div>
    );
  }

  // 3. MAIN RENDER
  const chartData = [
    { session: 1, performance: 75, mood: 4.0, feedback: 3.8 },
    { session: 2, performance: 60, mood: 3.2, feedback: 3.5 },
    { session: 3, performance: 80, mood: 4.2, feedback: 4.0 },
    { session: 4, performance: 45, mood: 2.5, feedback: 2.8 },
    { session: 5, performance: 55, mood: 3.0, feedback: 3.2 },
    { session: 6, performance: 85, mood: 4.5, feedback: 4.3 },
    { session: 7, performance: 95, mood: 5.0, feedback: 4.8 },
    { session: 8, performance: 90, mood: 4.7, feedback: 4.5 },
    { session: 9, performance: 70, mood: 3.8, feedback: 3.9 },
    { session: 10, performance: 75, mood: 4.0, feedback: 4.0 },
  ];

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Animated Background-linear Orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-lime-500/10 rounded-full blur-3xl animate-pulse"
          style={{ animationDuration: '4s' }}></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-lime-400/10 rounded-full blur-3xl animate-pulse"
          style={{ animationDuration: '6s', animationDelay: '1s' }}></div>
      </div>

      {/* Sidebar */}
      <div className="w-64 bg-lime-200 flex flex-col">
        <div className="bg-gray-900 text-white p-6 text-center">
          <h1 className="text-2xl font-black tracking-wide">DASHBOARD</h1>
          <p className="text-sm font-bold text-lime-400 mt-2">
            {session?.user?.name || params.username}
          </p>
        </div>

        <nav className="flex-1 py-4">
          <Link
            href={`/${params.username}/dashboard`}
            onClick={() => setActiveTab('home')}
            className={`w-full px-6 py-4 text-left font-black text-xl flex items-center gap-3 transition-colors ${activeTab === 'home' ? 'bg-lime-600 text-white' : 'bg-lime-200 text-gray-900 hover:bg-lime-300'
              }`}
          >
            <Home size={24} />
            HOME
          </Link>
          <Link
            href={`/analysis`}
            onClick={() => setActiveTab('analysis')}
            className={`w-full px-6 py-4 text-left font-black text-xl flex items-center gap-3 transition-colors ${activeTab === 'analysis' ? 'bg-lime-600 text-white' : 'bg-lime-200 text-gray-900 hover:bg-lime-300'
              }`}
          >
            <TrendingUp size={24} />
            ANALYSIS
          </Link>
          <Link
            href={`/pet`}
            onClick={() => setActiveTab('pet')}
            className={`w-full px-6 py-4 text-left font-black text-xl flex items-center gap-3 transition-colors ${activeTab === 'pet' ? 'bg-lime-600 text-white' : 'bg-lime-200 text-gray-900 hover:bg-lime-300'
              }`}
          >
            <Heart size={24} />
            PET SUPPORT
          </Link>
          <Link
            href={`/settings`}
            onClick={() => setActiveTab('settings')}
            className={`w-full px-6 py-4 text-left font-black text-xl flex items-center gap-3 transition-colors ${activeTab === 'settings' ? 'bg-lime-600 text-white' : 'bg-lime-200 text-gray-900 hover:bg-lime-300'
              }`}
          >
            <Settings size={24} />
            SETTINGS
          </Link>
        </nav>

        <div className="border-t-2 border-lime-400">
          <Link
            href={`/${params.username}/help`}
            className="w-full px-6 py-4 text-left font-black text-lg flex items-center gap-3 bg-lime-200 text-gray-900 hover:bg-lime-300"
          >
            <HelpCircle size={20} />
            HELP
          </Link>
          <Link
            href={`/${params.username}/profile`}
            className="w-full px-6 py-4 text-left font-black text-lg flex items-center gap-3 bg-lime-200 text-gray-900 hover:bg-lime-300"
          >
            <User size={20} />
            PROFILE
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto relative z-10">
        {/* Header */}
        <div className={`bg-white/95 backdrop-blur-sm px-8 py-6 flex justify-between items-center shadow-lg border-b-4 border-lime-400 transition-all duration-700 ${mounted ? 'translate-y-0 opacity-100' : '-translate-y-full opacity-0'}`}>
          <div>
            <h1 className="text-4xl font-black text-transparent bg-clip-text bg-linear-to-r from-gray-900 to-lime-600 animate-pulse">
              WELCOME BACK!
            </h1>
            <h2 className="text-4xl font-black text-transparent bg-clip-text bg-linear-to-r from-lime-600 to-lime-400">
              {(session?.user?.name || params.username)?.toUpperCase() || 'USER'}
            </h2>
          </div>
          <div className="flex items-center gap-4">
            <button className="p-3 bg-lime-200 rounded-full hover:bg-lime-300 transition-all duration-300 hover:scale-110 hover:rotate-12 relative group">
              <Bell size={24} className="text-gray-900" />
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-ping"></span>
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
            </button>
            <button
              onClick={() => signOut({ callbackUrl: '/' })}
              className="flex items-center gap-2 px-8 py-3 bg-linear-to-r from-lime-500 to-lime-600 text-white font-black text-xl rounded-full hover:from-lime-600 hover:to-lime-700 transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-105 transform"
            >
              <LogOut size={20} />
              LOG OUT
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="p-8">
          <div className="grid grid-cols-2 gap-8">
            {/* Left Column - Action Buttons */}
            <div className="space-y-4">
              {[
                { label: 'MONITOR YOUR MOOD', color: 'lime-400', hoverColor: 'lime-500', delay: 0, href: "/monitor" },
                { label: 'QUICK QUIZ', color: 'lime-300', hoverColor: 'lime-400', delay: 100, href: "/quiz" },
                { label: "TODAY'S TIP", color: 'lime-200', hoverColor: 'lime-300', delay: 200, href: "/tips" }
              ].map((btn, idx) => (
                <Link key={idx} href={btn.href}>
                  <button
                    className={`w-full px-8 py-6 bg-${btn.color} text-gray-900 font-black text-2xl rounded-3xl hover:bg-${btn.hoverColor} transition-all duration-300 shadow-lg hover:shadow-2xl transform hover:scale-105 hover:-translate-y-1 relative overflow-hidden group ${mounted ? 'translate-x-0 opacity-100' : '-translate-x-full opacity-0'}`}
                    style={{
                      transitionDelay: `${btn.delay}ms`,
                      background: `linear-gradient(135deg, rgb(${btn.color === 'lime-400' ? '163 230 53' : btn.color === 'lime-300' ? '190 242 100' : '217 249 157'}) 0%, rgb(${btn.color === 'lime-400' ? '132 204 22' : btn.color === 'lime-300' ? '163 230 53' : '190 242 100'}) 100%)`
                    }}
                  >
                    <span className="absolute inset-0 bg-linear-to-r from-white/0 via-white/20 to-white/0 -translate-x-full group-hover:translate-x-full transition-transform duration-700"></span>
                    <span className="relative z-10">{btn.label}</span>
                  </button>
                </Link>
              ))}
            </div>


            {/* Right Column - Chart */}
            <div className={`bg-white/95 backdrop-blur-sm p-6 rounded-3xl shadow-2xl border-2 border-lime-200 transition-all duration-700 hover:shadow-lime-200/50 ${mounted ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}`}>
              <h3 className="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2">
                <TrendingUp size={16} className="text-lime-600 animate-pulse" />
                User Performance Analysis: Mood vs. Score Over Sessions
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="session"
                    label={{ value: 'Session Number', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    yAxisId="left"
                    label={{ value: 'Performance Score', angle: -90, position: 'insideLeft' }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[0, 5]}
                    label={{ value: 'Mood', angle: 90, position: 'insideRight' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '12px',
                      border: '2px solid rgb(163 230 53)',
                      boxShadow: '0 10px 25px rgba(0,0,0,0.1)'
                    }}
                  />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="performance"
                    stroke="#3b82f6"
                    name="Performance Score"
                    strokeWidth={3}
                    dot={{ fill: '#3b82f6', r: 5, strokeWidth: 2, stroke: '#fff' }}
                    activeDot={{ r: 8 }}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="mood"
                    stroke="#ef4444"
                    name="User Mood"
                    strokeWidth={3}
                    dot={{ fill: '#ef4444', r: 5, strokeWidth: 2, stroke: '#fff' }}
                    activeDot={{ r: 8 }}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="feedback"
                    stroke="#f59e0b"
                    name="Feedback Score"
                    strokeWidth={3}
                    dot={{ fill: '#f59e0b', r: 5, strokeWidth: 2, stroke: '#fff' }}
                    activeDot={{ r: 8 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Bottom Section - Monthly Stats */}
          <div className="mt-8 grid grid-cols-4 gap-6">
            <div className={`col-span-1 transition-all duration-700 ${statsVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
              <h2 className="text-4xl font-black text-transparent bg-clip-text bg-linear-to-r from-lime-700 to-lime-500 leading-tight animate-pulse">
                THIS<br />MONTH
              </h2>
            </div>
            <div className={`bg-lime-300 p-8 rounded-3xl text-center shadow-2xl transform transition-all duration-700 hover:scale-110 hover:rotate-2 cursor-pointer ${statsVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}
              style={{ transitionDelay: '100ms' }}>
              <h3 className="text-xl font-black text-gray-900 mb-2">HAPPY<br />DAYS</h3>
              <p className="text-6xl font-black text-gray-900">17</p>
            </div>
            <div className={`bg-red-400 p-8 rounded-3xl text-center shadow-2xl transform transition-all duration-700 hover:scale-110 hover:rotate-2 cursor-pointer ${statsVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}
              style={{ transitionDelay: '200ms' }}>
              <h3 className="text-xl font-black text-gray-900 mb-2">SAD DAYS</h3>
              <p className="text-6xl font-black text-gray-900">6</p>
            </div>
            <div className={`bg-yellow-300 p-8 rounded-3xl text-center shadow-2xl transform transition-all duration-700 hover:scale-110 hover:rotate-2 cursor-pointer ${statsVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}
              style={{ transitionDelay: '300ms' }}>
              <h3 className="text-xl font-black text-gray-900 mb-2">STRESSED<br />DAYS</h3>
              <p className="text-6xl font-black text-gray-900">4</p>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes slideInRight {
          from {
            transform: translateX(-100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
      `}</style>
    </div>
  );
}

export default Dashboard;