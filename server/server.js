const express = require('express');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const PORT = 3000;
const HOST = '192.168.88.243';

// Middleware
app.use(cors());
app.use(bodyParser.json());

// File path for data.json
const dataFilePath = path.join(__dirname, 'data.json');

// Helper function to read and write JSON file
const readDataFile = () => {
  const rawData = fs.readFileSync(dataFilePath, 'utf8');
  return JSON.parse(rawData);
};

const writeDataFile = (data) => {
  fs.writeFileSync(dataFilePath, JSON.stringify(data, null, 2), 'utf8');
};

// Routes

// 1. Get all cameras
app.get('/cameras', (req, res) => {
  const data = readDataFile();
  res.json(data.cameras);
});

// 2. Get camera by name
app.get('/cameras/:name', (req, res) => {
  const { name } = req.params;
  const data = readDataFile();
  const camera = data.cameras.find((cam) => cam.camera_name === name);
  if (camera) {
    res.json(camera);
  } else {
    res.status(404).json({ message: 'Camera not found' });
  }
});

// 3. Add a new camera (Create)
app.post('/cameras', (req, res) => {
  const newCamera = req.body;
  const data = readDataFile();
  data.cameras.push(newCamera);
  writeDataFile(data);
  res.status(201).json({ message: 'Camera added successfully', newCamera });
});

// 4. Update a camera by name (Update)
app.put('/cameras/:name', (req, res) => {
  const { name } = req.params;
  const updatedData = req.body;

  const data = readDataFile();
  const cameraIndex = data.cameras.findIndex((cam) => cam.camera_name === name);

  if (cameraIndex !== -1) {
    data.cameras[cameraIndex] = { ...data.cameras[cameraIndex], ...updatedData };
    writeDataFile(data);
    res.json({ message: 'Camera updated successfully', updatedCamera: data.cameras[cameraIndex] });
  } else {
    res.status(404).json({ message: 'Camera not found' });
  }
});

// 5. Delete a camera by name (Delete)
app.delete('/cameras/:name', (req, res) => {
  const { name } = req.params;

  const data = readDataFile();
  const cameraIndex = data.cameras.findIndex((cam) => cam.camera_name === name);

  if (cameraIndex !== -1) {
    const deletedCamera = data.cameras.splice(cameraIndex, 1);
    writeDataFile(data);
    res.json({ message: 'Camera deleted successfully', deletedCamera });
  } else {
    res.status(404).json({ message: 'Camera not found' });
  }
});

// Start server
app.listen(PORT, HOST, () => {
  console.log(`Server is running on http://${HOST}:${PORT}`);
});
