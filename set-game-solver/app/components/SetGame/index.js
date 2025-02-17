import React from 'react';
import { 
  GameHeader, 
  ApiKeyInput, 
  UploadSection, 
  GameBoard, 
  SetsList 
} from './components';

const SetGame = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <GameHeader />
      <ApiKeyInput />
      <UploadSection />
      <GameBoard />
      <SetsList />
    </div>
  );
};

export default SetGame;