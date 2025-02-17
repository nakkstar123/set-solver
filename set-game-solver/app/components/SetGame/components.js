import React from 'react';
import { Upload, Eye } from 'lucide-react';

export const GameHeader = () => (
  <div className="text-center mb-12">
    <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600">
      SET Game Solver
    </h1>
    <p className="text-gray-400 text-lg">
      Upload a picture of SET cards and let AI find all possible sets
    </p>
  </div>
);

export const ApiKeyInput = () => (
  <div className="max-w-md mx-auto mb-8">
    <input
      type="password"
      placeholder="Enter your Claude API key"
      className="w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-700 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
    />
  </div>
);

export const UploadSection = () => (
  <div className="max-w-xl mx-auto mb-12">
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center">
        <Upload className="w-12 h-12 mx-auto mb-4 text-gray-500" />
        <p className="text-gray-400 mb-4">
          Drag and drop your SET game image here, or click to browse
        </p>
        <button className="px-6 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-gray-800">
          Choose File
        </button>
      </div>
    </div>
  </div>
);

export const GameBoard = () => {
  const mockCards = Array(12).fill(null).map((_, i) => ({
    id: i,
    imageUrl: `/images/00.png`
  }));

  return (
    <div className="mb-12">
      <h2 className="text-2xl font-semibold mb-6">Game Board</h2>
      {/* Responsive container with min/max width */}
      <div className="max-w-3xl mx-auto px-2">
        {/* Responsive grid container */}
        <div className="w-full sm:w-[480px] md:w-[600px] mx-auto">
          <div className="grid grid-cols-4 gap-1 sm:gap-2"> {/* Responsive gap */}
            {mockCards.map((card) => (
              <div
                key={card.id}
                className="relative transition-all duration-200 ease-in-out"
              >
                {/* Responsive card size */}
                <div className="w-[72px] sm:w-[110px] md:w-[140px] h-[96px] sm:h-[150px] md:h-[180px] rounded-lg overflow-hidden shadow-md">
                  <img
                    src={card.imageUrl}
                    alt={`Card ${card.id}`}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export const SetsList = () => {
  const mockSets = [
    { id: 1, cards: [0, 1, 2], explanation: "All red, different shapes" },
    { id: 2, cards: [3, 4, 5], explanation: "All striped, same number" },
    { id: 3, cards: [6, 7, 8], explanation: "All outlined, different colors" },
  ];

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-2xl font-semibold mb-6">Found Sets</h2>
      <div className="space-y-4">
        {mockSets.map((set) => (
          <div 
            key={set.id} 
            className="bg-gray-800 border border-gray-700 rounded-lg p-4 hover:bg-gray-750 transition-colors duration-200"
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium mb-2">
                  Set #{set.id}
                </h3>
                <p className="text-gray-400">{set.explanation}</p>
              </div>
              <button
                className="px-4 py-2 border border-purple-500 text-purple-500 rounded-lg hover:bg-purple-500 hover:text-white transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-gray-800 flex items-center gap-2"
              >
                <Eye className="w-4 h-4" />
                Highlight
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};