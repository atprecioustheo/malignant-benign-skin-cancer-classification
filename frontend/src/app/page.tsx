'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Cookies from 'js-cookie';
import { predictionAPI } from '@/lib/api';
import Dashboard from '@/components/Dashboard';
import Navigation from '@/components/Navigation';
import toast from 'react-hot-toast';

export default function Home() {
  const router = useRouter();
  const [dashboardData, setDashboardData] = useState({
    total: 0,
    benign: 0,
    malignant: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = Cookies.get('auth_token');
    if (!token) {
      router.push('/login');
      return;
    }

    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const data = await predictionAPI.getDashboard();
      setDashboardData(data);
    } catch (error) {
      toast.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">
            Melanoma Detection Dashboard
          </h1>
          <Dashboard data={dashboardData} />
        </div>
      </main>
    </div>
  );
}