var scene = new THREE.Scene();
scene.background = new THREE.Color( 0xcccccc );

var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;
camera.position.y = 5;
camera.position.x = 5;
camera.lookAt(new THREE.Vector3(0,0,0)); // Make the camera look at the point of origin


var renderer = new THREE.WebGLRenderer({antialias:true});
var devicePixelRatio = window.devicePixelRatio || 1; // To handle high pixel density displays

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(devicePixelRatio);

document.body.appendChild( renderer.domElement );





var render = function () {
 
 requestAnimationFrame( render );
 
 renderer.render(scene, camera);

};


 


// instantiate a texture loader
var loader = new THREE.TextureLoader();
//allow cross origin loading
loader.crossOrigin = '';



// The textures to use
var arr = [
 'https://s3-us-west-2.amazonaws.com/s.cdpn.io/259155/THREE_gates.jpg',
 'https://s3-us-west-2.amazonaws.com/s.cdpn.io/259155/THREE_crate1.jpg',
 'https://s3-us-west-2.amazonaws.com/s.cdpn.io/259155/THREE_crate2.jpg'
];
var textureToShow = 0;


// Load the first texture
// var texture = loadTexture('https://s3-us-west-2.amazonaws.com/s.cdpn.io/259155/MarbleSurface.jpg');

// Instantiate the material we will be using
var material = new THREE.MeshBasicMaterial();
// Instantiate a geometry to use
var geometry = new THREE.BoxGeometry( 1, 1, 1 );
// Instatiate the mesh with the geometry and material
var cube = new THREE.Mesh( geometry, material );
cube.position.y = 0.5;

// Then load the texture
loader.load(arr[textureToShow], function(tex) {
    // Once the texture has loaded
    // Asign it to the material
    material.map = tex;
    // Update the next texture to show
    textureToShow++;
    // Add the mesh into the scene
    scene.add( cube );
});




// Click interaction
var canvas = document.getElementsByTagName("canvas")[0];

canvas.addEventListener("click", function() {
  

 
 loader.load(arr[textureToShow], function(tex) {
  // Once the texture has loaded
  // Asign it to the material
  material.map = tex;
  // Update the next texture to show
  textureToShow++;
  // Have we got to the end of the textures array
  if(textureToShow > arr.length-1) {
   textureToShow = 0;
  }
 }); 
 
});




// Start rendering the scene
render();

