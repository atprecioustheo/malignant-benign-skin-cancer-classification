'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import { predictionAPI } from '@/lib/api';
import Navigation from '@/components/Navigation';
import toast from 'react-hot-toast';

export default function Upload() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [name, setName] = useState('');
  const [notes, setNotes] = useState('');
  // Using fixed voting model; no selection needed

  const onDrop = (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setFile(file);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      toast.error('Please select an image');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('name', name);
    formData.append('notes', notes);
    // No model_type needed; backend uses voting model only

    try {
      const response = await predictionAPI.predict(formData);
      setResult(response);
      toast.success('Prediction completed!');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setName('');
    setNotes('');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      <main className="max-w-4xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">
            Upload Image for Analysis
          </h1>

          {!result ? (
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Image Upload */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Image
                </label>
                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                    isDragActive 
                      ? 'border-blue-400 bg-blue-50' 
                      : 'border-gray-300 hover:border-gray-400'
                  }`}
                >
                  <input {...getInputProps()} />
                  {preview ? (
                    <div className="space-y-4">
                      <img
                        src={preview}
                        alt="Preview"
                        className="mx-auto h-64 w-64 object-cover rounded-lg"
                      />
                      <p className="text-sm text-gray-600">
                        Click or drag to replace image
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <svg
                        className="mx-auto h-12 w-12 text-gray-400"
                        stroke="currentColor"
                        fill="none"
                        viewBox="0 0 48 48"
                      >
                        <path
                          d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                          strokeWidth={2}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <p className="text-lg text-gray-600">
                        {isDragActive ? 'Drop the image here' : 'Drag & drop an image here, or click to select'}
                      </p>
                      <p className="text-sm text-gray-500">
                        PNG, JPG, JPEG up to 10MB
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Image Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Image Name (Optional)
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter a name for this image"
                />
              </div>

              {/* Model selection removed; using Voting model only */}

              {/* Notes */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Notes (Optional)
                </label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Add any notes about this image"
                />
              </div>

              {/* Submit Button */}
              <div className="flex space-x-4">
                <button
                  type="submit"
                  disabled={!file || loading}
                  className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? 'Analyzing...' : 'Analyze Image'}
                </button>
                <button
                  type="button"
                  onClick={reset}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                >
                  Reset
                </button>
              </div>
            </form>
          ) : (
            /* Results */
            <div className="bg-white rounded-lg shadow p-6 space-y-6">
              <h2 className="text-2xl font-bold text-gray-900">Analysis Results</h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Uploaded Image</h3>
                  {preview && (
                    <img
                      src={preview}
                      alt="Analyzed"
                      className="w-full h-64 object-cover rounded-lg"
                    />
                  )}
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold">Prediction</h3>
                    <div className={`text-2xl font-bold ${
                      result.result === 'Benign' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {result.result}
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold">Confidence</h3>
                    <div className="text-xl text-gray-800">{result.confidence}</div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold">Model Used</h3>
                    <div className="text-gray-600">voting</div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold">Timestamp</h3>
                    <div className="text-gray-600">
                      {new Date(result.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="flex space-x-4">
                <button
                  onClick={reset}
                  className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors"
                >
                  Analyze Another Image
                </button>
                <button
                  onClick={() => router.push('/history')}
                  className="border border-gray-300 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-50 transition-colors"
                >
                  View History
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}