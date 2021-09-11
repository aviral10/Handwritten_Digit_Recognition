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
    })   
}
loadModel()

// Setup Sketchpad
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
//     document
//   .getElementById('canv')
//   .scrollIntoView({ behavior: 'smooth' });
    let ctx = sk.getContext('2d');
    let imageData = ctx.getImageData(0, 0, sk.width, sk.height);
    predict_multi(imageData)
    // console.log(imageData)
}

function clear_canvas(){
    let ctx = sk.getContext('2d')
    ctx.fillStyle = "#FFFFFF"
    ctx.fillRect(0, 0, 300, 300)
}
clear_canvas()

let fakeCanvas = document.createElement('Canvas')
function tensorToImage(arr, width, height){
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
    
    ctx = fakeCanvas.getContext('2d');
    fakeCanvas.width = width;
    fakeCanvas.height = height;
    // create imageData object
    let idata = ctx.createImageData(width, height);
    // set our buffer as source
    idata.data.set(buffer);
    // update fakeCanvas with new data
    ctx.putImageData(idata,0,0);
    // create a new img object
    let image=new Image();
    // set the img.src to the fakeCanvas data url
    image.src=fakeCanvas.toDataURL();
    // append the new img object to the page
    return image
}


let images = []
let layersLabels = [[28,1,1],
                    [26,64,8],
                    [24,64,8],
                    [12,64,8],
                    [12,64,8],
                    [10,128,8],
                    [8, 128,8],
                    [4, 128,8],
                    [4, 128,8],
                    [2, 256,16],
                    [1, 256,16],
                    [1,15,15],
                    [1,30,30],
                    [1,30,30],
                    [1,10,10]]
let finalLabels  = [[26,64,8],
                    [24,64,8],
                    [12,64,8],
                    [10,128,8],
                    [4, 128,8],
                    [2, 256,16],
                    [1, 256,16],
                    [1, 15,15],
                    [1, 30,30],
                    [1, 10, 10]]


function predict_multi(im){
    
    images = []
    let tensor = tf.tidy(() => {
        let ts = tf.browser.fromPixels(im, 1);
        ts = tf.cast(ts,'float32');
        ts = ts.div(tf.scalar(-255))
        ts = ts.add(tf.scalar(1))
        ts = tf.image.resizeBilinear(ts, [28,28]).mean(2).expandDims(-1).expandDims()
        return ts;
    });
    tf.engine().startScope()
    for(let i=1;i<15;i++){
        if(i == 3 || i == 6 ||  i == 7 || i == 12) continue
        
        // console.log("I: ", i)
        
        let modelA = layers[i];
        let prediction = modelA.predict(tensor);
        // prediction.print(true)
        if(i >= 11){
            let synced = prediction.dataSync()
            
            if(i == 14){
                synced = synced.map((num)=>{
                    return (num*100).toPrecision(4)
                })
                updateChart(synced);
            }
            images.push(synced)
            continue;
        }
        
        let shape = layersLabels[i][0]
        let layer_images = []
        for(let k = 0;k<layersLabels[i][1];k++){
            let synced = prediction.slice([0,0,0,k], [1,shape,shape,1]).reshape([shape, shape]).dataSync()
            let img = tensorToImage(synced, shape, shape)
            layer_images.push(img)
        }
        images.push(layer_images)
        
    }
    for(let i=0;i<images.length-3;i++){
        applyTexturesbyLayer(i);
    }

    tensor.dispose();
    tf.engine().endScope()
    console.log("Tensors: ", tf.memory().numTensors);
    console.log(images)
}


function predict_basic(im){
    let tensor = tf.tidy(() => {
        let ts = tf.browser.fromPixels(im, 1);
        ts = tf.cast(ts,'float32');
        ts = ts.div(tf.scalar(-255))
        ts = ts.add(tf.scalar(1))
        ts = tf.image.resizeBilinear(ts, [28,28]).mean(2).expandDims(-1).expandDims()
        return ts;
    });
    
    tf.engine().startScope()
    let prediction = layers[14].predict(tensor);
    let preds = prediction.dataSync().map((num)=>{
        return (num*100).toPrecision(4)
    })
    updateChart(preds);
    tf.engine().endScope()
    tensor.dispose();
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
updateChart([0,0,0,0,0,0,0,0,0,0]);





/// THREE JS 
import * as dat from 'dat.gui'
import { update } from '@tensorflow/tfjs-layers/dist/variables';
import { random } from 'gsap/all';

const group = new THREE.Group();
// Canvas
const canvas = document.querySelector('canvas.webgl')

// Scene
const scene = new THREE.Scene()

/**
 * Object
 */

// let gui = new dat.GUI()


let textureLoader = new THREE.TextureLoader()
let texture = textureLoader.load('texture_none.png')
texture.magFilter = THREE.NearestFilter


let ratio = 1/14
let gap = 1
// const geometry = new THREE.BoxBufferGeometry(28*ratio, 28*ratio, 0.026)

// const mesh = new THREE.Mesh(geometry, material)
// scene.add(mesh)

let meshes = []

let gaps = [5,5,2,2,1.5,1.5,1.5,1.5,1.5,1.5]
function generateMeshes(){
    for(let i=1;i<15;i++){
        if(i == 3 || i == 6 || i == 7 || i == 12) continue
        
        let I = layersLabels[i][2]
        let J = layersLabels[i][1]/layersLabels[i][2]
        let TEMP = []
        let sz = layersLabels[i][0]
        for(let ii=0;ii<I;ii++){
            let temp = []
            for(let jj = 0;jj<J;jj++){
                let geometry, mesh
                let material = new THREE.MeshBasicMaterial({ map: texture })
                if(I == 1 || J == 1){
                    geometry = new THREE.BoxBufferGeometry(0.2, 0.2, 0.03)
                    mesh = new THREE.Mesh(geometry, material)
                    // let gap = 0.5
                    mesh.position.x = (ii*gap)+ii - I;
                    mesh.position.y = (jj*gap)+jj - J;
                    mesh.position.z = -((i*5) - i) ;
                }else{
                    geometry = new THREE.BoxBufferGeometry(sz*ratio, sz*ratio, 0.03)
                    mesh = new THREE.Mesh(geometry, material)
                    mesh.position.x = (ii*gap)+ii - I;
                    mesh.position.y = (jj*gap)+jj - J;
                    mesh.position.z = -((i*5) - i) ;
                }
                
                temp.push(mesh)
                // scene.add(mesh)
                group.add(mesh)
            }
            TEMP.push(temp)
        }
        meshes.push(TEMP)
    }
}
generateMeshes()



function generateALine(posA, posB){
    const materialA = new THREE.LineBasicMaterial( { color: 0xffffff } );
    // const materialB = new THREE.LineBasicMaterial( { color: 0x000000 } );
    materialA.transparent = true;
    materialA.opacity = 0.2
    const points = [];
    points.push( posA );
    points.push( posB );

    const geometry = new THREE.BufferGeometry().setFromPoints( points );

    const line = new THREE.Line( geometry, materialA );
    // scene.add( line );
    group.add(line)
}

function randomRange(min, max) {
    return Math.floor(Math.random() * (max - min) + min);
}

function generateLines(){
    for(let layer=0;layer<finalLabels.length-3-1;layer++){
        let I = finalLabels[layer][2]
        let J = finalLabels[layer][1]/finalLabels[layer][2]
        for(let i=0;i<I;i+=randomRange(1,3)){
            for(let j=0;j<J;j+=randomRange(1,3)){
                let II = finalLabels[layer+1][2]
                let JJ = finalLabels[layer+1][1]/finalLabels[layer+1][2]
                for(let k=1;k<II;k+=randomRange(3,5)){
                    for(let l=1;l<JJ;l+=randomRange(3,5)){
                        generateALine(meshes[layer][i][j].position, meshes[layer+1][k][l].position);
                        // console.log(i,j,k,l)
                    }
                }
            }
        }
    }
    for(let layer=finalLabels.length-3-1;layer<finalLabels.length-3;layer++){
        let I = finalLabels[layer][2]
        let J = finalLabels[layer][1]/finalLabels[layer][2]
        for(let i=0;i<I;i+=randomRange(1,3)){
            for(let j=0;j<J;j+=randomRange(1,3)){
                let II = finalLabels[layer+1][2]
                let JJ = finalLabels[layer+1][1]/finalLabels[layer+1][2]
                for(let k=0;k<II;k++){
                    for(let l=0;l<JJ;l++){
                        generateALine(meshes[layer][i][j].position, meshes[layer+1][k][l].position);
                        // console.log(i,j,k,l)
                    }
                }
            }
        }
    }
    for(let layer=finalLabels.length-3;layer<finalLabels.length-1;layer++){
        let I = finalLabels[layer][2]
        let J = finalLabels[layer][1]/finalLabels[layer][2]
        for(let i=0;i<I;i+=2){
            for(let j=0;j<J;j+=1){
                let II = finalLabels[layer+1][2]
                let JJ = finalLabels[layer+1][1]/finalLabels[layer+1][2]
                for(let k=0;k<II;k+=randomRange(3,5)){
                    for(let l=0;l<JJ;l+=randomRange(3,5)){
                        generateALine(meshes[layer][i][j].position, meshes[layer+1][k][l].position);
                        // console.log(i,j,k,l)
                    }
                }
            }
        }
        // console.log(I,J)
    }

}

function getRandomArray(n){
    let random = []
    for(let i=0;i<n;i++){
        random.push(i)
    }
    random = random.sort(() => .5 - Math.random()).slice(0,randomRange(0,Math.min(100,n))).sort()
    return random
}
// function generateLines(){
//     console.log(meshes)
//     for(let layer=0;layer<meshes.length-1;layer++){
//         let I = finalLabels[layer][2]
//         let J = finalLabels[layer][1]/finalLabels[layer][2]
//         let ri = getRandomArray(I)
//         for(let i of ri){
//             let rj = getRandomArray(J)
//             for(let j of rj){
//                 let II = finalLabels[layer+1][2]
//                 let JJ = finalLabels[layer+1][1]/finalLabels[layer+1][2]
//                 let rii = getRandomArray(II)
//                 for(let k of rii){
//                     let rjj = getRandomArray(JJ)
//                     for(let l of rjj){
                        
//                         generateALine(meshes[layer][i][j].position, meshes[layer+1][k][l].position);
//                         // console.log(i,j,k,l)
//                     }
//                 }
//             }
//         }
//     }
// }
generateLines()

function applyTexturesbyLayer(layer){
    let I = finalLabels[layer][2]
    let J = finalLabels[layer][1]/finalLabels[layer][2]
    let sz = finalLabels[layer][0]
    for(let i=0;i<I;i++){
        for(let j=0;j<J;j++){
            let num = i*J+j;
            // console.log("Layer: ", layer)
            let image = images[layer][num]
            image.onload = ()=>{
                // console.log(num)
                updateTexture(image, meshes[layer][i][j].material)
            }
        }
    }
}


let mesh = meshes[1][0][0]
// gui.add(mesh.position, 'x').min(mesh.position.x).max(mesh.position.x+3).step(0.001)
// gui.add(mesh.position, 'y').min(mesh.position.y).max(mesh.position.y+3).step(0.001)
// gui.add(mesh.position, 'z').min(mesh.position.z).max(mesh.position.z+3).step(0.001)

function updateTexture(image, material) {
    texture = new THREE.Texture( image );
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    texture.generateMipmaps = false
    texture.needsUpdate = true;
    material.map = texture
};

const sizee = 10;
const divisions = 10;

const gridHelper = new THREE.GridHelper( sizee, divisions );
scene.add( gridHelper );


// let amb = new THREE.AmbientLight(0xffffff, 0.5)
// let pointLight = new THREE.PointLight(0x555555, 0.5)
// pointLight.position.x = 2
// pointLight.position.y = 3
// pointLight.position.z = 4
// const sphereSize = 1;
// const pointLightHelper = new THREE.PointLightHelper( pointLight, sphereSize );
// scene.add( pointLightHelper );
// scene.add(amb, pointLight)
// const geo = new THREE.TorusBufferGeometry(0.3, 0.2, 16, 32)
// let mater = new THREE.MeshPhongMaterial()
// mater.transparent = true
// mater.opacity = 0.5
// mater.shininess = 100
// let meshi = new THREE.Mesh(geo, mater)
// meshi.position.x = 1.5
// scene.add(meshi)



/**
 * Sizes
 */
const sizes = {
    width: window.innerWidth,
    height: window.innerHeight,
    // width: 1000,
    // height: 700
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
const camera = new THREE.PerspectiveCamera(45, sizes.width / sizes.height, 0.1, 500)
camera.position.x = 0
camera.position.y = 40
camera.position.z = 50
scene.add(camera)

// Controls
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
// controls.target = new THREE.Vector3(meshes[0][3][3].position.x, meshes[0][3][3].position.y,meshes[0][3][3].position.z)
controls.target = new THREE.Vector3(0,0,0)

THREE.Object3D.prototype.rotateAroundWorldAxis = function() {

    var q1 = new THREE.Quaternion();
    return function ( point, axis, angle ) {

        q1.setFromAxisAngle( axis, angle );

        this.quaternion.multiplyQuaternions( q1, this.quaternion );

        this.position.sub( point );
        this.position.applyQuaternion( q1 );
        this.position.add( point );

        return this;
    }

}();

let p = new THREE.Vector3(0, 0, 0);
let ax = new THREE.Vector3(0, 0.5, 0);



new THREE.Box3().setFromObject( group ).getCenter( group.position ).multiplyScalar( - 1 );

// scene.add( object );
scene.add(group)
// gui.add(p, 'x').min(-50).max(50).step(0.01)
// gui.add(p, 'y').min(-50).max(50).step(0.01)
// gui.add(p, 'z').min(-50).max(50).step(0.01)
// gui.add(ax, 'x').min(-50).max(50).step(0.01)
// gui.add(ax, 'y').min(-50).max(50).step(0.01)
// gui.add(ax, 'z').min(-50).max(50).step(0.01)



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
    // group.rotation.y = elapsedTime
    // Render
    group.rotateAroundWorldAxis(p, ax, 0.008);
    renderer.render(scene, camera)

    // Call tick again on the next frame
    window.requestAnimationFrame(tick)
}

tick()