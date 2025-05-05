import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export const detectObjects = async (imageData: string) => {
  try {
    const response = await axios.post(`${API_URL}/api/detect`, {
      image: imageData
    });
    return response.data;
  } catch (error) {
    console.error('Error detecting objects:', error);
    throw error;
  }
}; 