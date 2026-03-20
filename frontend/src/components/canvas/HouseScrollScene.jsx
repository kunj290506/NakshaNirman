import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Text, Environment, ContactShadows, Grid, Edges } from '@react-three/drei'
import * as THREE from 'three'

function Room({ position, size, color, label, targetScroll, progress, cinematic }) {
    const meshRef = useRef()
    const textRef = useRef()

    useFrame((_, delta) => {
        const dist = Math.abs(progress - targetScroll)
        const revealBand = 0.2
        const activity = 1 - clamp01(dist / revealBand)

        const targetY = 0.18 + activity * (cinematic ? 0.9 : 0.75)
        meshRef.current.position.y = THREE.MathUtils.lerp(meshRef.current.position.y, targetY, delta * 5)
        meshRef.current.scale.y = THREE.MathUtils.lerp(meshRef.current.scale.y, 0.65 + activity * 0.95, delta * 5)

        meshRef.current.material.emissiveIntensity = THREE.MathUtils.lerp(
            meshRef.current.material.emissiveIntensity,
            activity * (cinematic ? 0.7 : 0.45),
            delta * 5
        )

        if (textRef.current) {
            textRef.current.fillOpacity = THREE.MathUtils.lerp(
                textRef.current.fillOpacity,
                0.06 + activity * 0.96,
                delta * 5
            )
            textRef.current.position.y = meshRef.current.position.y + 0.6
            const scale = 0.92 + activity * 0.25
            textRef.current.scale.setScalar(scale)
        }
    })

    return (
        <group position={[position[0], 0, position[2]]}>
            <mesh ref={meshRef} receiveShadow castShadow>
                <boxGeometry args={[size[0], 0.5, size[2]]} />
                <meshStandardMaterial
                    color={color}
                    emissive={color}
                    emissiveIntensity={0}
                    roughness={0.28}
                    metalness={0.1}
                />
                <Edges color="#2a2a2a" threshold={15} />
            </mesh>
            <Text
                ref={textRef}
                position={[0, 1, 0]}
                rotation={[-Math.PI / 2, 0, 0]}
                fontSize={0.5}
                color="#18181b"
                anchorX="center"
                anchorY="middle"
                fillOpacity={0.1}
            >
                {label}
            </Text>
        </group>
    )
}

export default function HouseScrollScene({ progress = 0, cinematic = false }) {
    const groupRef = useRef()
    const scanRef = useRef()
    const cameraPath = useMemo(() => {
        const points = cinematic
            ? [[-1.8, 7.2, 7.2], [2.1, 6.8, 6.1], [1.2, 6.1, 5.1], [-2.2, 5.7, 4.8], [0.6, 5.3, 4.1]]
            : [[-0.8, 7.8, 8.1], [1.3, 7.2, 7.1], [0.2, 6.8, 6.2], [-1.1, 6.5, 5.8], [0.4, 6.2, 5.3]]
        return new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)))
    }, [cinematic])

    const lookPath = useMemo(() => {
        const points = cinematic
            ? [[0.2, 0.2, 0.4], [-0.2, 0.35, 0.1], [0.0, 0.4, -0.4], [0.1, 0.45, -1.3], [0.0, 0.45, -2.0]]
            : [[0.0, 0.2, 0.2], [0.1, 0.3, -0.2], [-0.1, 0.35, -0.7], [0.0, 0.4, -1.2], [0.0, 0.42, -1.6]]
        return new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p)))
    }, [cinematic])

    useFrame((state, delta) => {
        const offset = clamp01(progress)

        if (groupRef.current) {
            groupRef.current.rotation.y = THREE.MathUtils.lerp(
                groupRef.current.rotation.y,
                (-0.35 + offset * 1.6) * Math.PI,
                delta * 3
            )

            groupRef.current.rotation.x = THREE.MathUtils.lerp(
                groupRef.current.rotation.x,
                -0.22 + Math.sin(offset * Math.PI * 2) * 0.015,
                delta * 3
            )

            groupRef.current.position.y = THREE.MathUtils.lerp(
                groupRef.current.position.y,
                -0.82 + offset * 1.35,
                delta * 3
            )
        }

        const cameraTarget = cameraPath.getPoint(offset)
        state.camera.position.lerp(cameraTarget, delta * 2.3)

        const lookTarget = lookPath.getPoint(offset)
        state.camera.lookAt(lookTarget)

        if (scanRef.current) {
            scanRef.current.position.z = THREE.MathUtils.lerp(scanRef.current.position.z, 3.2 - offset * 7.2, delta * 5)
            scanRef.current.material.opacity = 0.06 + smoothStep(0.12, 0.85, offset) * 0.16
        }
    })

    const rooms = [
        { id: 'living', label: 'Living Room', pos: [0, 0, 2], size: [4, 0.5, 3], color: '#f0f0f0', scroll: 0.15 },
        { id: 'kitchen', label: 'Kitchen', pos: [-3, 0, -1], size: [3, 0.5, 3], color: '#e6e6e6', scroll: 0.35 },
        { id: 'master', label: 'Master Br.', pos: [2.5, 0, -1], size: [3.5, 0.5, 3], color: '#dddddd', scroll: 0.55 },
        { id: 'bath', label: 'Bathroom', pos: [-2, 0, -3.5], size: [2, 0.5, 2], color: '#d4d4d4', scroll: 0.75 },
        { id: 'bed2', label: 'Bedroom', pos: [2, 0, -3.5], size: [3, 0.5, 2], color: '#cbcbcb', scroll: 0.95 },
    ]

    return (
        <group ref={groupRef} position={[0, -1, 0]}>
            <ambientLight intensity={0.52} />
            <directionalLight position={[10, 20, 10]} intensity={1.25} castShadow shadow-mapSize={[2048, 2048]} />
            <directionalLight position={[-12, 8, -8]} intensity={0.4} color="#d0d0d0" />
            <Environment preset={cinematic ? 'sunset' : 'city'} />

            <Grid
                position={[0, -0.04, -0.45]}
                args={[16, 16]}
                cellColor="#0b1220"
                sectionColor="#3d3d3d"
                cellSize={0.5}
                sectionSize={2}
                cellThickness={0.5}
                sectionThickness={1}
                fadeDistance={30}
                fadeStrength={1}
                infiniteGrid={false}
            />

            <mesh receiveShadow position={[0, -0.05, -0.5]}>
                <boxGeometry args={[11, 0.1, 10]} />
                <meshStandardMaterial color={cinematic ? '#eaf2ff' : '#ffffff'} roughness={1} />
            </mesh>

            <mesh ref={scanRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.07, 2.8]}>
                <planeGeometry args={[10.8, 1.4]} />
                <meshBasicMaterial color="#8a8a8a" transparent opacity={0.12} />
            </mesh>

            {rooms.map(room => (
                <Room
                    key={room.id}
                    position={room.pos}
                    size={room.size}
                    color={room.color}
                    label={room.label}
                    targetScroll={room.scroll}
                    progress={offsetClamp(progress)}
                    cinematic={cinematic}
                />
            ))}

            <ContactShadows position={[0, -0.1, 0]} opacity={0.6} scale={20} blur={2.5} far={4} />
        </group>
    )
}

function offsetClamp(value) {
    return Math.max(0, Math.min(1, value))
}

function clamp01(value) {
    return Math.max(0, Math.min(1, value))
}

function smoothStep(edge0, edge1, x) {
    const t = clamp01((x - edge0) / (edge1 - edge0))
    return t * t * (3 - 2 * t)
}
