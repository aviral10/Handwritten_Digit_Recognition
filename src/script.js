import * as tf from '@tensorflow/tfjs';
import * as THREE from 'three';
import { TextBufferGeometry } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Sketchpad } from './sketchpad_new.js';
import './style.css';

/// prep

let model;
let layers = [];
let answer;
let base_image = []

// Load the model and Initialize respective layers of the model
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


// Setup Sketchpad to take the user input
let sk = document.getElementById('sketchpad')
let sketchpad = new Sketchpad({
    element: '#sketchpad',
    width: 300,
    height: 300
});
sketchpad.penSize = 40
let fakeCanvas = document.createElement('Canvas')


// Setting up buttons
let butt = document.getElementById('clr')
butt.onclick = clear_canvas
butt = document.getElementById('threeDbutt')
butt.onclick = callmemaybe


let choice = 0;
// Make it Realtime
sk.onmousemove = function(event) {
    if(event.buttons == 1) {
        callmemaybe();
    }
}
sk.ontouchmove = function(event) {
    callmemaybe();
}


// Scroll to the results
let dont_render_lines = false;
let threeDbutt = document.getElementById('threeDbutt')
threeDbutt.onclick = ()=>{
    let w = parseInt(window.innerWidth);
    if(choice == 0){
        choice = 1;
        sk.onmousemove = ()=>{};
        
        threeDbutt.innerHTML = "Predict"
        document.getElementById("canv_holder").style.display = 'block';
        
        // For mobile devices
        if(w <= 800) {
            dont_render_lines = true;
        }
        threeD();
    }else{
        
        callmemaybe();    
    }
    if(w <= 800) {
        return;
    }
    $('html, body').animate({
        'scrollTop' : $("#canv").position().top
    }, 1000);
}


function callmemaybe(){
    let ctx = sk.getContext('2d');
    let imageData = ctx.getImageData(0, 0, sk.width, sk.height);
    if(choice == 1)
        predict_multi(imageData)
    else 
        predict_basic(imageData)
        // console.log(imageData)
}


function clear_canvas(){
    let ctx = sk.getContext('2d')
    ctx.fillStyle = "#FFFFFF"
    ctx.fillRect(0, 0, 300, 300)
}
clear_canvas()


function tensorToImage(arr, width, height){
    let buffer = new Uint8ClampedArray(width * height * 4);   // have enough bytes
    for(let y = 0; y < height; y++) {
        for(let x = 0; x < width; x++) {
            let POS = (y * width + x)
            let pos = POS * 4;                                // position in buffer based on x and y
            buffer[pos  ] = 255 - Math.floor(arr[POS]*255);   // some R value [0, 255]
            buffer[pos+1] = 255 - Math.floor(arr[POS]*255);   // some G value
            buffer[pos+2] = 255 - Math.floor(arr[POS]*255);   // some B value
            buffer[pos+3] = 255;                              // set alpha channel
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
                    [1,15,15],
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
    // Initialize a sandbox environment for the new tensors
    let tensor = tf.tidy(() => {
        let ts = tf.browser.fromPixels(im, 1);
        // Invert the image
        ts = tf.cast(ts,'float32');
        ts = ts.div(tf.scalar(-255))
        ts = ts.add(tf.scalar(1))
        // ts = tf.image.resizeBilinear(ts, [28,28]).mean(2).expandDims(-1).expandDims()
        ts = tf.image.resizeBilinear(ts, [28,28]).mean(2)
    
        base_image = ts.dataSync();
        ts = ts.expandDims(-1).expandDims()
        return ts;
    });
    tf.engine().startScope()
    for(let i=1;i<15;i++){
        if(i == 3 || i == 6 ||  i == 7 || i == 11) continue         // Ignore these layers
        let modelA = layers[i];
        let prediction = modelA.predict(tensor);
        // Populate different layers with actual outputs of the intermediate layers
        if(i >= 11){
            let synced = prediction.dataSync()
            // If we are in the last layer, calculate the final answer
            if(i == 14){
                answer = prediction.argMax(1).dataSync()[0]
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
    
    // Update textures in the 3D visualization
    for(let i=0;i<images.length;i++){
        applyTexturesbyLayer(i);
    }
    update_base()
    images = []
    tensor.dispose();
    tf.engine().endScope()

    // Logging Number of Tensors
    // console.log("Tensors: ", tf.memory().numTensors);
}


function predict_basic(im){
    let tensor = tf.tidy(() => {
        let ts = tf.browser.fromPixels(im, 1);
        ts = tf.cast(ts,'float32');
        // invert the image as: -x + 1
        ts = ts.div(tf.scalar(-255))
        ts = ts.add(tf.scalar(1))
        ts = tf.image.resizeBilinear(ts, [28,28]).mean(2).expandDims(-1).expandDims()
        return ts;
    });
    
    tf.engine().startScope()
    // Output of the last layer
    let prediction = layers[14].predict(tensor);
    let preds = prediction.dataSync().map((num)=>{
        return (num*100).toPrecision(4)
    })
    updateChart(preds);
    tf.engine().endScope()
    tensor.dispose();
    // Logging Number of Tensors
    // console.log("Tensors: ", tf.memory().numTensors);
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
        // responsive: false,
        maintainAspectRatio: false,
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



// Setting up a new group
const group = new THREE.Group();
// Canvas
const canvas = document.querySelector('canvas.webgl')

// Scene
const scene = new THREE.Scene()

// for debug purposes
// let gui = new dat.GUI()


let textureLoader = new THREE.TextureLoader()
let texture = textureLoader.load('texture_none.png')
texture.magFilter = THREE.NearestFilter

let fontLoader = new THREE.FontLoader()


// Preparing 3D Scene
let ratio = 1/14
let gap = 1
let meshes = []
let nums_t = 0
let gaps = [5,5,2,2,1.5,1.5,1.5,1.5,1.5,1.5]
let text_array = []
let base_meshes = []


function generateMeshes(){
    for(let i=1;i<15;i++){
        if(i == 3 || i == 6 || i == 7 || i == 11) continue
        
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
                    if(I == 10 ||  J == 10){
                        fontLoader.load(
                            '/fonts/helvetiker_regular.typeface.json',
                            (font)=>{
                                let textGeometry = new TextBufferGeometry(
                                    String(nums_t++),
                                    {
                                        font: font,
                                        size: 1,
                                        height: 0.2,
                                        curveSegments: 6,
                                        bevelEnabled: true,
                                        bevelThickness: 0.03,
                                        bevelSize: 0.02,
                                        bevelOffset: 0,
                                        bevelSegments: 5
                                    }
                                )
                                let textMaterial = new THREE.MeshBasicMaterial()
                                let text = new THREE.Mesh(textGeometry, textMaterial);
                                text.position.x = (ii*gap)+ii - I;
                                text.position.y = (jj*gap)+jj - J;
                                text.position.z = -((i*5) - i) ;
                                text.rotation.x = Math.PI
                                text.rotation.z = Math.PI
                                text_array.push(text)
                                group.add(text);
                            }
                        )
                    }
                }else{
                    geometry = new THREE.BoxBufferGeometry(sz*ratio, sz*ratio, 0.03)
                    mesh = new THREE.Mesh(geometry, material)
                    mesh.position.x = (ii*gap)+ii - I;
                    mesh.position.y = (jj*gap)+jj - J;
                    mesh.position.z = -((i*5) - i) ;
                }
                
                temp.push(mesh)
                group.add(mesh)
            }
            TEMP.push(temp)
        }
        meshes.push(TEMP)
    }
    // Generate meshes for the base layer as well
    base_meshes = []
    let sz = layersLabels[0][0]
    for(let i=0;i<28;i++){
        for(let j=0;j<28;j++){
            let geometry = new THREE.BoxBufferGeometry(8*ratio, 8*ratio, 0.1)
            let material = new THREE.MeshBasicMaterial()
            let mesh = new THREE.Mesh(geometry, material)
            mesh.position.x = (i*0.05)+i - 15;
            mesh.position.y = (j*0.05)+j - 15;
            mesh.position.z = 5;
            base_meshes.push(mesh)
            group.add(mesh)
        }
    }
}
generateMeshes()


function generateALine(posA, posB){
    // Create a material 
    const materialA = new THREE.LineBasicMaterial( { color: 0xffffff } );
    // Set properties for the material
    materialA.transparent = true;
    materialA.opacity = 0.2
    const points = [];
    points.push( posA );
    points.push( posB );
    //Create geometry for the object
    const geometry = new THREE.BufferGeometry().setFromPoints( points );
    const line = new THREE.Line( geometry, materialA );
    group.add(line)
}

// Utitlity function for getting a random number between a given range
function randomRange(min, max) {
    return Math.floor(Math.random() * (max - min) + min);
}


function generateLines(){
    // Draw the connections between all the layers
    if(dont_render_lines == false){
        // Dont execute on mobile devices
        for(let layer=0;layer<10-3-1;layer++){
            // This logic works for the first 7 layers
            let I = finalLabels[layer][2]
            let J = finalLabels[layer][1]/finalLabels[layer][2]
            for(let i=0;i<I;i+=randomRange(1,3)){
                for(let j=0;j<J;j+=randomRange(1,3)){
                    let II = finalLabels[layer+1][2]
                    let JJ = finalLabels[layer+1][1]/finalLabels[layer+1][2]
                    for(let k=1;k<II;k+=randomRange(3,5)){
                        for(let l=1;l<JJ;l+=randomRange(3,5)){
                            generateALine(meshes[layer][i][j].position, meshes[layer+1][k][l].position);
                        }
                    }
                }
            }
        }
    }
    // Further layers are handled here
    for(let layer=10-3;layer<10-1;layer++){
        let I = finalLabels[layer][2]
        let J = finalLabels[layer][1]/finalLabels[layer][2]
        for(let i=0;i<I;i+=2){
            for(let j=0;j<J;j+=1){
                let II = finalLabels[layer+1][2]
                let JJ = finalLabels[layer+1][1]/finalLabels[layer+1][2]
                for(let k=0;k<II;k+=randomRange(3,5)){
                    for(let l=0;l<JJ;l+=randomRange(3,5)){
                        generateALine(meshes[layer][i][j].position, meshes[layer+1][k][l].position);
                    }
                }
            }
        }
    }
}


function getRandomArray(n){
    // Utility function for generating a random array
    let random = []
    for(let i=0;i<n;i++){
        random.push(i)
    }
    random = random.sort(() => .5 - Math.random()).slice(0,randomRange(0,Math.min(100,n))).sort()
    return random
}


function update_base(){
    // Handling the base layer seperately
    // updating textures for the base 28x28 image
    for(let i=0;i<28;i++){
        for(let j=0;j<28;j++){
            let val = Math.floor(Math.abs(base_image[i*28+j]*255));
            let texture = new THREE.MeshBasicMaterial();
            texture.color = new THREE.Color("rgb("+val+","+val+","+val+")")
            base_meshes[j*28+27-i].material = texture
            texture.needsUpdate = true;
        }
    }
    // base_meshes = []
}


function applyTexturesbyLayer(layer){
    // Update textures of all the layers when a new prediction is made
    // Works on a single layer for the param:layer
    if(layer >= 7){
        let sz = finalLabels[layer][1]
        for(let i=0;i<sz;i++){
            let val = Math.floor(Math.abs(images[layer][i]*255));
            let texture = new THREE.MeshBasicMaterial();
            texture.color = new THREE.Color("rgb("+val+","+val+","+val+")")
            meshes[layer][i][0].material = texture
            if(layer == 10-1){
                text_array[i].material = texture
            }
            texture.needsUpdate = true;
        }
        return;
    }
    // Execute the below logic for layers 1-6
    let I = finalLabels[layer][2]
    let J = finalLabels[layer][1]/finalLabels[layer][2]
    let sz = finalLabels[layer][0]
    for(let i=0;i<I;i++){
        for(let j=0;j<J;j++){
            let num = i*J+j;
            let image = images[layer][num]
            // Update textures when the image is fully loaded
            image.onload = ()=>{
                updateTexture(image, meshes[layer][i][j].material)
            }
        }
    }
}


function updateTexture(image, material) {
    // Update texture of a single image based on the material recieved as a parameter
    texture = new THREE.Texture( image );
    texture.magFilter = THREE.NearestFilter
    texture.minFilter = THREE.NearestFilter
    texture.generateMipmaps = false
    texture.needsUpdate = true;
    material.map = texture
};


// Sizes
const sizee = 10;
const divisions = 10;

// Debug
// const gridHelper = new THREE.GridHelper( sizee, divisions );
// scene.add( gridHelper );

const sizes = {
    width: window.innerWidth,
    height: window.innerHeight,
    // width: 1000,
    // height: 700
}

window.addEventListener('resize', () => {
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
camera.position.y = 0
camera.position.z = 50
scene.add(camera)

// Controls
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true

// Origin 
controls.target = new THREE.Vector3(0,0,0)

// Utility function for setting up the rotation of the entire scene
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

// Setting up a pivot point for the rotation
let p = new THREE.Vector3(0, 0, 0);
let ax = new THREE.Vector3(0, 1, 0);
new THREE.Box3().setFromObject( group ).getCenter( group.position ).multiplyScalar( - 1 );
let  pivot = new THREE.Group();
pivot.add( group );
group.position.set( 0, 0, 28 );
scene.add( pivot );
// scene.add(group)

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
let rotateAnim = true;
const clock = new THREE.Clock()
const tick = () => {
    const elapsedTime = clock.getElapsedTime()

    // Update controls
    controls.update()
    // group.rotation.y = elapsedTime
    
    // Render
    if(rotateAnim)
        pivot.rotation.y = elapsedTime*0.2
    // group.rotateAroundWorldAxis(p, ax, 0.008);
    renderer.render(scene, camera)

    // Call tick again on the next frame
    window.requestAnimationFrame(tick)
}


function threeD(){ 
    generateLines()
    callmemaybe();
    tick()
}


function toggleRotation(){
    rotateAnim = !rotateAnim
}


document.addEventListener("keypress", function onEvent(event) {
    if (event.key === "r") {
        toggleRotation();
    }
});