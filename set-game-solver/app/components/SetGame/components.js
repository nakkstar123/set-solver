"use client";
import React, { useState } from 'react';
import { Upload, Eye } from 'lucide-react';

const SET_COLORS = [
  'rgb(239 68 68)', // red
  'rgb(34 197 94)', // green
  'rgb(59 130 246)', // blue
  'rgb(168 85 247)', // purple
  'rgb(234 179 8)', // yellow
  'rgb(236 72 153)', // pink
];

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

export const GameBoard = ({ selectedSet, setColor }) => {
  const mockCards = Array(12).fill(null).map((_, i) => ({
    id: i,
    imageUrl: "images/00.png"
  }));

  return (
    <div className="mb-12">
      <h2 className="text-2xl font-semibold mb-6">Game Board</h2>
      <div className="max-w-3xl mx-auto px-2">
        <div className="w-full sm:w-[480px] md:w-[600px] mx-auto">
          <div className="grid grid-cols-4 gap-1 sm:gap-2">
            {mockCards.map((card) => {
              const isHighlighted = selectedSet?.includes(card.id);
              return (
                <div
                  key={card.id}
                  className="relative transition-all duration-200 ease-in-out"
                >
                  <div 
                    className={`
                      w-[72px] sm:w-[110px] md:w-[140px] 
                      h-[96px] sm:h-[150px] md:h-[180px] 
                      rounded-lg overflow-hidden shadow-md
                      transition-all duration-200
                      ${isHighlighted ? 'scale-105' : ''}
                    `}
                    style={{
                      boxShadow: isHighlighted ? `0 0 0 3px ${setColor}, 0 4px 6px -1px rgb(0 0 0 / 0.1)` : undefined
                    }}
                  >
                    <img
                      src={card.imageUrl}
                      alt={`Card ${card.id}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export const SetsList = ({ onSetSelected }) => {
  const [activeSetId, setActiveSetId] = useState(null);
  
  const mockSets = [
    { id: 1, cards: [0, 1, 2] },
    { id: 2, cards: [3, 4, 5] },
    { id: 3, cards: [6, 7, 8] },
    { id: 4, cards: [9, 10, 11] },
  ];

  const handleSetClick = (set) => {
    if (activeSetId === set.id) {
      setActiveSetId(null);
      onSetSelected(null, null);
    } else {
      setActiveSetId(set.id);
      onSetSelected(set.cards, SET_COLORS[(set.id - 1) % SET_COLORS.length]);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-2xl font-semibold mb-6 text-center">Found Sets</h2>
      <div className="flex flex-wrap gap-3 justify-center items-center">
        {mockSets.map((set) => {
          const setColor = SET_COLORS[(set.id - 1) % SET_COLORS.length];
          const isActive = activeSetId === set.id;
          return (
            <button
              key={set.id}
              onClick={() => handleSetClick(set)}
              className={`
                px-4 py-2 rounded-lg font-medium
                transition-all duration-200
                focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800
              `}
              style={{
                backgroundColor: isActive ? setColor : 'transparent',
                color: isActive ? 'white' : setColor,
                border: `2px solid ${setColor}`,
              }}
            >
              Set {set.id}
            </button>
          );
        })}
      </div>
    </div>
  );
};