const video = document.getElementById('video');
let lastProcessingTime = 0;
const PROCESSING_THROTTLE = 1000;
let cachedVideoPredictions = null;
const BASE_IMAGE_PATH = '/images/';
let model;
let currentStream = null;
let compareInterval;

const IMAGE_GROUPS = {
    group1: ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'],
    group2: ['img6.jpg', 'img7.jpg', 'img8.jpg', 'img9.jpg', 'img10.jpg'],
    group3: ['img11.jpg', 'img12.jpg', 'img13.jpg', 'img14.jpg', 'img15.jpg']
};

// Crear elementos de imagen ocultos
const loadedImages = {};
Object.entries(IMAGE_GROUPS).forEach(([groupName, images]) => {
    loadedImages[groupName] = [];
    images.forEach(imgSrc => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = BASE_IMAGE_PATH + imgSrc;
        loadedImages[groupName].push(img);
    });
});

const cameraSelect = document.getElementById('cameraSelect');

async function loadModel() {
    try {
        model = await mobilenet.load({
            version: 2,
            alpha: 1.0
        });
        console.log('Modelo cargado exitosamente');
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
    }
}

async function listCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        cameraSelect.innerHTML = '<option value="">Seleccionar cámara...</option>';
        videoDevices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Cámara ${cameraSelect.length}`;
            cameraSelect.appendChild(option);
        });
    } catch (err) {
        console.error('Error al enumerar dispositivos:', err);
    }
}

function stopCurrentStream() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    if (compareInterval) {
        clearInterval(compareInterval);
    }
}

async function startCamera(deviceId = '') {
    stopCurrentStream();
    const constraints = {
        video: {
            width: { ideal: 300 },
            height: { ideal: 225 },
            ...(deviceId && { deviceId: { exact: deviceId } })
        }
    };
    
    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = currentStream;
        await video.play();
        if (!deviceId) {
            await listCameras();
        }
        compareInterval = setInterval(compareImages, 1000);
    } catch (err) {
        console.error('Error al acceder a la cámara:', err);
    }
}

document.getElementById('startCamera').addEventListener('click', () => {
    startCamera(cameraSelect.value);
});

cameraSelect.addEventListener('change', () => {
    if (cameraSelect.value) {
        startCamera(cameraSelect.value);
    }
});

async function compareImages() {
    if (!model || !currentStream) return;
    
    const now = Date.now();
    if (now - lastProcessingTime < PROCESSING_THROTTLE) return;
    lastProcessingTime = now;

    try {
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 225;
        const ctx = canvas.getContext('2d');
        
        ctx.drawImage(video, 0, 0, 300, 225);
        const videoPredictions = await model.classify(canvas, 10);
        
        Object.entries(loadedImages).forEach(async ([groupName, images]) => {
            const groupIndex = parseInt(groupName.replace('group', ''));
            
            const similarities = await Promise.all(images.map(async (img, i) => {
                if (!img.complete) return 0;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, 300, 225);
                const imagePredictions = await model.classify(canvas, 10);
                const similarity = comparePredictions(videoPredictions, imagePredictions);
                
                const similarityDiv = document.getElementById(`g${groupIndex}-similarity${i + 1}`);
                similarityDiv.textContent = `G${groupIndex} - Similitud ${i + 1}: ${(similarity * 100).toFixed(2)}%`;
                similarityDiv.className = `similarity ${similarity > 0.7 ? 'high' : ''}`;
                
                return similarity;
            }));

            const totalSimilarityDiv = document.getElementById(`g${groupIndex}-totalSimilarity`);
            const validSimilarities = similarities.filter(s => s > 0);
            if (validSimilarities.length > 0) {
                const average = validSimilarities.reduce((a, b) => a + b) / validSimilarities.length;
                totalSimilarityDiv.textContent = `Similitud Total Grupo ${groupIndex}: ${(average * 100).toFixed(2)}%`;
                totalSimilarityDiv.className = `group-total-similarity ${average > 0.7 ? 'high' : ''}`;
            }
        });
    } catch (error) {
        console.error('Error al comparar imágenes:', error);
    }
}

function comparePredictions(pred1, pred2) {
    const map1 = new Map(pred1.map(p => [p.className, p.probability]));
    const map2 = new Map(pred2.map(p => [p.className, p.probability]));

    const topClasses = [...new Set([...map1.keys(), ...map2.keys()].slice(0, 10))];

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (const className of topClasses) {
        const prob1 = map1.get(className) || 0;
        const prob2 = map2.get(className) || 0;
        const weight = prob1 > 0 && prob2 > 0 ? 1.2 : 1;
        dotProduct += prob1 * prob2 * weight;
        norm1 += prob1 * prob1;
        norm2 += prob2 * prob2;
    }

    return norm1 && norm2 ? Math.max(0, Math.min(1, dot
