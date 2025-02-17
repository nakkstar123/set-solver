"use client";
import React, { useState } from 'react';
import { 
  GameHeader, 
  ApiKeyInput, 
  UploadSection, 
  GameBoard, 
  SetsList 
} from './components';

const SetGame = () => {
  const [selectedSet, setSelectedSet] = useState(null);
  const [activeColor, setActiveColor] = useState(null);

  const handleSetSelected = (cards, color) => {
    setSelectedSet(cards);
    setActiveColor(color);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <GameHeader />
      <ApiKeyInput />
      <UploadSection />
      <GameBoard selectedSet={selectedSet} setColor={activeColor} />
      <SetsList onSetSelected={handleSetSelected} />
    </div>
  );
};

export default SetGame;