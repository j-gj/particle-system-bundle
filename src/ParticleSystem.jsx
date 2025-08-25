import React, { useMemo, useState, useRef, useEffect } from 'react'
import { Canvas, createPortal, useFrame, extend, useThree } from '@react-three/fiber'
import { OrbitControls, useFBO } from '@react-three/drei'
import * as THREE from 'three'

// Utility function
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16) / 255,
    g: parseInt(result[2], 16) / 255,
    b: parseInt(result[3], 16) / 255
  } : null;
}

function getRandomSphere(count, size) {
  const data = new Float32Array(count * 4)
  for (let i = 0; i < count * 4; i += 4) {
    data[i] = (Math.random() - 0.5) * 4
    data[i + 1] = (Math.random() - 0.5) * 4
    data[i + 2] = (Math.random() - 0.5) * 4
    data[i + 3] = 1
  }
  return data
}

// Shader noise functions
const simplexNoise = `
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }
`

const snoiseVec3 = `
vec3 snoiseVec3(vec3 x) {
  float s  = snoise(vec3( x ));
  float s1 = snoise(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
  float s2 = snoise(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
  vec3 c = vec3( s , s1 , s2 );
  return c;
}
`

const curlNoise = `
vec3 curlNoise(vec3 p) {
  const float e = .1;
  vec3 dx = vec3( e   , 0.0 , 0.0 );
  vec3 dy = vec3( 0.0 , e   , 0.0 );
  vec3 dz = vec3( 0.0 , 0.0 , e   );

  vec3 p_x0 = snoiseVec3( p - dx );
  vec3 p_x1 = snoiseVec3( p + dx );
  vec3 p_y0 = snoiseVec3( p - dy );
  vec3 p_y1 = snoiseVec3( p + dy );
  vec3 p_z0 = snoiseVec3( p - dz );
  vec3 p_z1 = snoiseVec3( p + dz );

  float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  const float divisor = 1.0 / ( 2.0 * e );
  return normalize( vec3( x , y , z ) * divisor );
}
`

// Optimized Classic Perlin (cnoise) - similar to yours, but unified helpers
const classicNoise = `
  vec3 fade(vec3 t) { return t*t*t*(t*(t*6.0-15.0)+10.0); }
  float cnoise(vec3 P) { // Your version is already close; use this if testing
    vec3 Pi0 = floor(P);
    vec3 Pi1 = Pi0 + vec3(1.0);
    Pi0 = mod289(Pi0);
    Pi1 = mod289(Pi1);
    vec3 Pf0 = fract(P);
    vec3 Pf1 = Pf0 - vec3(1.0);
    vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
    vec4 iy = vec4(Pi0.yy, Pi1.yy);
    vec4 iz0 = Pi0.zzzz;
    vec4 iz1 = Pi1.zzzz;
    vec4 ixy = permute(permute(ix) + iy);
    vec4 ixy0 = permute(ixy + iz0);
    vec4 ixy1 = permute(ixy + iz1);
    vec4 gx0 = ixy0 / 7.0;
    vec4 gy0 = fract(floor(gx0) / 7.0) - 0.5;
    gx0 = fract(gx0);
    vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
    vec4 sz0 = step(gz0, vec4(0.0));
    gx0 -= sz0 * (step(0.0, gx0) - 0.5);
    gy0 -= sz0 * (step(0.0, gy0) - 0.5);
    vec4 gx1 = ixy1 / 7.0;
    vec4 gy1 = fract(floor(gx1) / 7.0) - 0.5;
    gx1 = fract(gx1);
    vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
    vec4 sz1 = step(gz1, vec4(0.0));
    gx1 -= sz1 * (step(0.0, gx1) - 0.5);
    gy1 -= sz1 * (step(0.0, gy1) - 0.5);
    vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
    vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
    vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
    vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
    vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
    vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
    vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
    vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);
    vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
    g000 *= norm0.x; g010 *= norm0.y; g100 *= norm0.z; g110 *= norm0.w;
    vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
    g001 *= norm1.x; g011 *= norm1.y; g101 *= norm1.z; g111 *= norm1.w;
    float n000 = dot(g000, Pf0);
    float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
    float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
    float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
    float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
    float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
    float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
    float n111 = dot(g111, Pf1);
    vec3 fade_xyz = fade(Pf0);
    vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
    vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
    float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
    return 2.2 * n_xyz;
  }

`

// Simulation Material
class SimulationMaterial extends THREE.ShaderMaterial {
  constructor(size = 512) {
    const positionsTexture = new THREE.DataTexture(
      getRandomSphere(size * size, 1),
      size,
      size,
      THREE.RGBAFormat,
      THREE.FloatType
    )
    positionsTexture.needsUpdate = true

    super({
      uniforms: {
        positions: { value: positionsTexture },
        uFrequency: { value: 0.25 },
        uTime: { value: 0 }
      },
      vertexShader: `
        precision mediump float;
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        precision mediump float;
        precision mediump sampler2D;
        uniform float uTime;
        uniform float uFrequency;
        uniform sampler2D positions;
        varying vec2 vUv;

        ${simplexNoise}
        ${snoiseVec3}
        ${curlNoise}
        ${classicNoise}

        void main() {
          float time = uTime * 0.015;
          vec3 pos = texture2D(positions, vUv).rgb;
          vec3 curlPos = texture2D(positions, vUv).rgb;

          pos = curlNoise(pos * uFrequency + time);
          curlPos = curlNoise(curlPos * uFrequency + time);
          curlPos += curlNoise(curlPos * uFrequency * 2.0) * 0.5;
          curlPos += curlNoise(curlPos * uFrequency * 4.0) * 0.25;
          curlPos += curlNoise(curlPos * uFrequency * 8.0) * 0.125;

          gl_FragColor = vec4(mix(pos, curlPos, snoise(pos + time) * 0.5 + 0.5), 1.0);
        }
      `
    })
  }
}

// Depth of Field Material
class DepthOfFieldMaterial extends THREE.ShaderMaterial {
  constructor() {
    super({
      uniforms: {
        positions: { value: null },
        pointSize: { value: 3 },
        uTime: { value: 0 },
        uFocus: { value: 4 },
        uFov: { value: 45 },
        uBlur: { value: 30 },
        uGradientColors: { value: new Float32Array([1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]) },
        uGradientStops: { value: new Float32Array([0.0, 0.3, 0.7, 1.0]) },
        uGradientRadius: { value: 2.0 }
      },
      vertexShader: `
        precision mediump float;
        uniform sampler2D positions;
        uniform float pointSize;
        uniform float uTime;
        uniform float uFocus;
        uniform float uFov;
        uniform float uBlur;
        uniform float uGradientRadius;
        varying float vDistance;
        varying float vGradientDistance;
        varying vec3 vWorldPosition;

        void main() {
          vec3 pos = texture2D(positions, position.xy).xyz;
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          vec4 worldPosition = modelMatrix * vec4(pos, 1.0);

          gl_Position = projectionMatrix * mvPosition;

          vDistance = abs(uFocus - -mvPosition.z);
          vGradientDistance = length(worldPosition.xyz) / uGradientRadius;

          float sizeFactor = step(1.0 - (1.0 / uFov), position.x);
          gl_PointSize = sizeFactor * vDistance * uBlur;
        }
      `,
      fragmentShader: `
        precision mediump float;
        varying float vDistance;
        varying float vGradientDistance;
        varying vec3 vWorldPosition;
        uniform vec3 uGradientColors[4];
        uniform float uGradientStops[4];
        uniform float uTime;

        vec3 getGradientColor(float t) {
          t = clamp(t, 0.0, 1.0);
          vec3 color = mix(uGradientColors[0], uGradientColors[1], smoothstep(uGradientStops[0], uGradientStops[1], t));
          color = mix(color, uGradientColors[2], smoothstep(uGradientStops[1], uGradientStops[2], t));
          color = mix(color, uGradientColors[3], smoothstep(uGradientStops[2], uGradientStops[3], t));
          return color;
        }

        void main() {
          vec2 cxy = 2.0 * gl_PointCoord - 1.0;
          float r2 = dot(cxy, cxy);
          if (r2 > 1.0) discard;
          float mask = 1.0 - smoothstep(0.95, 1.0, r2);

          float alpha = (1.04 - clamp(vDistance, 0.0, 1.0)) * mask;

          float timeOffset = sin(uTime * 0.5) * 0.1;
          vec3 gradientColor = getGradientColor(vGradientDistance + timeOffset);

          gl_FragColor = vec4(gradientColor, alpha);
        }
      `,
      transparent: true,
      blending: THREE.NormalBlending,
      depthWrite: false
    })
  }
}

// Extend materials
extend({ SimulationMaterial, DepthOfFieldMaterial })

// Particles Component
function Particles({ 
  frequency = 0.15,
  speedFactor = 4, 
  fov = 35, 
  blur = 24, 
  focus = 8.7,
  size = 256,
  gradientColors = ['#F0F4FF', '#637AFF', '#372CD5', '#F0F4FF'],
  gradientStops = [0.6, 0.65, 0.75, 0.8],
  gradientRadius = 1.35,
  ...props 
}) {
  const simRef = useRef()
  const renderRef = useRef()
  
  // Set up FBO scene
  const [scene] = useState(() => new THREE.Scene())
  const [camera] = useState(() => new THREE.OrthographicCamera(-1, 1, 1, -1, 1 / Math.pow(2, 53), 1))
  const [positions] = useState(() => new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0]))
  const [uvs] = useState(() => new Float32Array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]))
  
  const target = useFBO(size, size, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat,
    stencilBuffer: false,
    type: THREE.FloatType
  })
  
  // Generate particle positions as UV coordinates
  const particles = useMemo(() => {
    const length = size * size
    const particles = new Float32Array(length * 3)
    
    for (let i = 0; i < length; i++) {
      const i3 = i * 3
      particles[i3 + 0] = (i % size) / size
      particles[i3 + 1] = Math.floor(i / size) / size  
      particles[i3 + 2] = 0
    }

    return particles
  }, [size])
  
  // Convert gradient colors to uniform format
  const gradientData = useMemo(() => {
    const colors = gradientColors.map(color => {
      const rgb = hexToRgb(color);
      return [rgb.r, rgb.g, rgb.b];
    });
    
    return {
      colors: new Float32Array(colors.flat()),
      stops: new Float32Array(gradientStops)
    };
  }, [gradientColors, gradientStops]);
  
  // Update simulation every frame
  useFrame(({ gl, clock }) => {
    if (!simRef.current || !renderRef.current) return
    
    // Render simulation to FBO
    gl.setRenderTarget(target)
    gl.clear()
    gl.render(scene, camera)
    gl.setRenderTarget(null)
    
    // Update render material with type assertion
    const renderMaterial = renderRef.current
    if (renderMaterial && renderMaterial.uniforms) {
      renderMaterial.uniforms.positions.value = target.texture
      renderMaterial.uniforms.uFocus.value = focus
      renderMaterial.uniforms.uFov.value = fov
      renderMaterial.uniforms.uBlur.value = blur
      renderMaterial.uniforms.uGradientColors.value = gradientData.colors
      renderMaterial.uniforms.uGradientStops.value = gradientData.stops
      renderMaterial.uniforms.uGradientRadius.value = gradientRadius
      renderMaterial.uniforms.uTime.value = clock.elapsedTime
    }
    
    // Update simulation material with type assertion
    const simMaterial = simRef.current
    if (simMaterial && simMaterial.uniforms) {
      simMaterial.uniforms.uTime.value = clock.elapsedTime * speedFactor
      simMaterial.uniforms.uFrequency.value = THREE.MathUtils.lerp(
        simMaterial.uniforms.uFrequency.value, 
        frequency, 
        0.1
      )
    }
  })
  
  return (
    <>
      {/* Simulation mesh rendered to FBO */}
      {createPortal(
        <mesh>
          <simulationMaterial ref={simRef} args={[size]} />
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={positions.length / 3} array={positions} itemSize={3} />
            <bufferAttribute attach="attributes-uv" count={uvs.length / 2} array={uvs} itemSize={2} />
          </bufferGeometry>
        </mesh>,
        scene
      )}
      
      {/* Points using FBO texture for positions */}
      <points {...props}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={particles.length / 3} array={particles} itemSize={3} />
        </bufferGeometry>
        <depthOfFieldMaterial ref={renderRef} />
      </points>
    </>
  )
}

// Internal App Component
function App({
  backgroundColor = '#fff', // Add backgroundColor prop
  frequency = 0.15,
  speedFactor = 4,
  rotationSpeed = 3.3,
  gradientColors = ['#F0F4FF', '#637AFF', '#372CD5', '#F0F4FF'],
  gradientStops = [0.6, 0.65, 0.75, 0.8],
  gradientRadius = 1.35,
  autoRotate = true,
  enableVerticalRotation = true,
  blur = 24,
  focus = 8.7,
  fov = 35,
  cameraZ = 7.6,
  particles = 256
}) {
  const { camera, gl, size } = useThree()
  const controlsRef = useRef()

  // Set renderer clear color
  useEffect(() => {
    const color = new THREE.Color(backgroundColor);
    gl.setClearColor(color, 1);
  }, [gl, backgroundColor]);
  

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
    //   camera.aspect = window.innerWidth / window.innerHeight
        camera.aspect = size.width / size.height
      camera.updateProjectionMatrix()
      //gl.setSize(window.innerWidth, window.innerHeight)
      gl.setSize(size.width, size.height)
      //let Canvas handle DPR
    //   gl.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    }
    window.addEventListener('resize', handleResize)
    handleResize()
    return () => window.removeEventListener('resize', handleResize)
  }, [camera, gl, size])

  // Use useFrame to update controls
  useFrame((state, delta) => {
    if (controlsRef.current && controlsRef.current.update) {
      controlsRef.current.update(delta)
    }
  })

  // Update camera position
  useEffect(() => {
    camera.position.set(0, 0, cameraZ)
  }, [cameraZ, camera])

  return (
    <>
      <OrbitControls
        ref={controlsRef}
        makeDefault
        autoRotate={autoRotate}
        autoRotateSpeed={rotationSpeed}
        enableZoom={false}
        enableDamping={true}
        dampingFactor={0.05}
        enableRotate={true}
        minPolarAngle={enableVerticalRotation ? 0 : Math.PI / 2}
        maxPolarAngle={enableVerticalRotation ? Math.PI : Math.PI / 2}
      />
      <ambientLight />
      <Particles
        frequency={frequency}
        speedFactor={speedFactor}
        fov={fov}
        blur={blur}
        focus={focus}
        position={[0, 0, 0]}
        size={particles}
        gradientColors={gradientColors}
        gradientStops={gradientStops}
        gradientRadius={gradientRadius}
      />
    </>
  )
}

/**
 * @framerSupportedLayoutWidth any
 * @framerSupportedLayoutHeight any
 * @framerIntrinsicWidth 200
 * @framerIntrinsicHeight 200
 */
export default function ParticleSystem({
  backgroundColor = '#fff',
  frequency = 0.15,
  speedFactor = 4,
  rotationSpeed = 0.3,
  gradientColors = ['#F0F4FF', '#637AFF', '#372CD5', '#F0F4FF'],
  gradientStops = [0.6, 0.65, 0.75, 0.8],
  gradientRadius = 1.35,
  autoRotate = true,
  enableVerticalRotation = false,
  blur = 24,
  focus = 8.7,
  fov = 35,
  cameraZ = 7.6,
  particles = 256,
}) {
  return (
    <Canvas
        camera={{
          fov: fov,
          position: [0, 0, cameraZ]
        }}
        gl={{
          alpha: false, // Keep alpha false for slight performance benefit
          antialias: true,
          powerPreference: "high-performance",
          desynchronized: true,
          premultipliedAlpha: false,
          preserveDrawingBuffer: false,
          failIfMajorPerformanceCaveat: false,
          stencil: false,
          depth: true
        }}
        resize={{ scroll: false }}
        dpr={[1, 2]}
        style={{ 
            position: "absolute",
            top: 0,
            left: 0,
            width: '100%', 
            height: '100%', 
            display: 'block',
            minWidth: '200px',
            minHeight: '200px',
            background: backgroundColor // Fallback for non-WebGL scenarios
        }}
      >
        <App 
          backgroundColor={backgroundColor} // Pass backgroundColor to App
          frequency={frequency}
          speedFactor={speedFactor}
          rotationSpeed={rotationSpeed}
          gradientColors={gradientColors}
          gradientStops={gradientStops}
          gradientRadius={gradientRadius}
          autoRotate={autoRotate}
          enableVerticalRotation={enableVerticalRotation}
          blur={blur}
          focus={focus}
          fov={fov}
          cameraZ={cameraZ}
          particles={particles}
        />
      </Canvas>
  )
}