import './style.css'
import './sketchpad.js'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import * as tf from '@tensorflow/tfjs';
// import { update } from '@tensorflow/tfjs-layers/dist/variables';

/// prep

let model;
let layers = [];

function loadModel(){
    model = tf.loadLayersModel("mnist_model_js/model.json");
    model.then((res)=>{
        for(let i=0;i<15;i++){
            let temp = tf.sequential();
            temp.add(res.layers[i]);
            layers.push(temp);
        }
        console.log(layers)
    })   
}
loadModel()


let sk = document.getElementById('sketchpad')
let sketchpad = new Sketchpad({
    element: '#sketchpad',
    width: 300,
    height: 300
});
sketchpad.penSize = 40

// Setting up buttons
let butt = document.getElementById('clr')
butt.onclick = clear_canvas
butt = document.getElementById('pred')
pred.onclick = callmemaybe

// Make it Realtime
sk.onmousemove = function(event) {
    if(event.buttons == 1) {
        // callmemaybe();
    }
}

function callmemaybe(){
    let ctx = sk.getContext('2d');
    let imageData = ctx.getImageData(0, 0, sk.width, sk.height);
    predict(imageData)
    // console.log(imageData)
}

function clear_canvas(){
    let ctx = sk.getContext('2d')
    ctx.fillStyle = "#FFFFFF"
    ctx.fillRect(0, 0, 300, 300)
}
clear_canvas()

function predict(im){
    // tf.engine().startScope()

    let tensor = tf.tidy(() => {
        let ts = tf.browser.fromPixels(im, 1);
        ts = tf.cast(ts,'float32');
        ts = ts.div(tf.scalar(-255))
        ts = ts.add(tf.scalar(1))
        ts = tf.image.resizeBilinear(ts, [28,28]).mean(2).expandDims(-1).expandDims()
        let tempo = layers[2].predict(ts)
        tempo.print(true)
        return ts;
    });

    
    model.then(function (res) {
        tf.engine().startScope()
        let prediction = res.predict(tensor);
        let xx = prediction.argMax(1).dataSync();
        console.log("Prediction: ", xx[0]);

        let preds = prediction.dataSync().map((num)=>{
            return (num*100).toPrecision(4)
        })
        // console.log(preds)
        updateChart(preds);

        let m1 = tf.sequential();
        m1.add(res.layers[1])
        let yy = m1.predict(tensor);
        
        let yyy = yy.slice([0,0,0,0],[1,26,26,1]).reshape([26,26]).dataSync()
        // console.log(yyy)
        matToImg(yyy, 26, 26)

        // display(yyy)

        tf.engine().endScope()
        tensor.dispose();
    }); 
    console.log("Tensors: ", tf.memory().numTensors);
}


/// Chart
var ctx = document.getElementById('myChart').getContext('2d');
Chart.defaults.global.defaultFontSize = 15;
var myChart = new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['0','1','2','3','4','5','6','7','8','9'],
        datasets: [{
            label: 'Confidence %',
            data: [0,0,0,0,0,0,0,0,0,0],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        responsive: false,
        scales: {
            yAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: 'Confidence plot for the predictions'
                },
                ticks: {
                    beginAtZero: true,
                    // steps: 10,
                    // stepValue: 5,
                    max: 100
                }
            }],
            xAxes: [{
                scaleLabel: {
                    display: true,
                    labelString: 'Digits'
                }
            }]
            },

    }
});

function updateChart(vals){
    // Updating the chart
    myChart.data.datasets[0].data = vals
    myChart.update()
}
updateChart([0,0,0,0,100,0,0,0,0,0]);










/// THREE JS 
import * as dat from 'dat.gui'


// Canvas
const canvas = document.querySelector('canvas.webgl')

// Scene
const scene = new THREE.Scene()

/**
 * Object
 */

let gui = new dat.GUI()


let textureLoader = new THREE.TextureLoader()
let texture = textureLoader.load('texture_4.png')
texture.magFilter = THREE.NearestFilter



const geometry = new THREE.BoxBufferGeometry(1, 1, 1)
const material = new THREE.MeshBasicMaterial({ map: texture })
const mesh = new THREE.Mesh(geometry, material)
scene.add(mesh)
gui.add(mesh.scale, 'x').min(0).max(3).step(0.001)
gui.add(mesh.scale, 'y').min(0).max(3).step(0.001)
gui.add(mesh.scale, 'z').min(0).max(3).step(0.001)



function matToImg(arr, width, height){
    let buffer = new Uint8ClampedArray(width * height * 4); // have enough bytes
    for(let y = 0; y < height; y++) {
        for(let x = 0; x < width; x++) {
            let POS = (y * width + x)
            let pos = POS * 4; // position in buffer based on x and y
            
            buffer[pos  ] = 255 - Math.floor(arr[POS]*255);           // some R value [0, 255]
            buffer[pos+1] = 255 - Math.floor(arr[POS]*255);           // some G value
            buffer[pos+2] = 255 - Math.floor(arr[POS]*255);           // some B value
            buffer[pos+3] = 255;           // set alpha channel
        }
    }
    let canvas = document.createElement('canvas'),
    ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    // create imageData object
    let idata = ctx.createImageData(width, height);
    // set our buffer as source
    idata.data.set(buffer);
    // update canvas with new data
    ctx.putImageData(idata,0,0);
    // create a new img object
    let image=new Image();
    // set the img.src to the canvas data url
    image.src=canvas.toDataURL();
    // append the new img object to the page
    
    let ii = document.getElementById('sample')
    ii.src = image.src
    ii.onload = ()=>{
        updateTexture(ii, material)
    }
}

function updateTexture(image, material) {
    texture = new THREE.Texture( image );
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    texture.generateMipmaps = false
    texture.needsUpdate = true;
    material.map = texture
};






/**
 * Sizes
 */
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight
}

window.addEventListener('resize', () =>
{
    // Update sizes
    sizes.width = window.innerWidth
    sizes.height = window.innerHeight

    // Update camera
    camera.aspect = sizes.width / sizes.height
    camera.updateProjectionMatrix()

    // Update renderer
    renderer.setSize(sizes.width, sizes.height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
})


/**
 * Camera
 */
// Base camera
const camera = new THREE.PerspectiveCamera(75, sizes.width / sizes.height, 0.1, 100)
camera.position.x = 1
camera.position.y = 1
camera.position.z = 1
scene.add(camera)

// Controls
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true

/**
 * Renderer
 */
const renderer = new THREE.WebGLRenderer({
    canvas: canvas
})
renderer.setSize(sizes.width, sizes.height)
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))

/**
 * Animate
 */
const clock = new THREE.Clock()

const tick = () =>
{
    const elapsedTime = clock.getElapsedTime()

    // Update controls
    controls.update()

    // Render
    renderer.render(scene, camera)

    // Call tick again on the next frame
    window.requestAnimationFrame(tick)
}

tick()