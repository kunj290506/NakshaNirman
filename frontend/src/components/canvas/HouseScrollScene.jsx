import { useRef, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { Text, Environment, ContactShadows } from '@react-three/drei'
import * as THREE from 'three'

function Room({ position, size, color, label, targetScroll }) {
    const meshRef = useRef()
    const textRef = useRef()

    useFrame((state, delta) => {
        const scrollOffset = state.scene.userData.scrollProgress || 0;
        
        const dist = Math.abs(scrollOffset - targetScroll)
        const isActive = dist < 0.15

        const targetY = isActive ? 0.75 : 0.25
        meshRef.current.position.y = THREE.MathUtils.lerp(meshRef.current.position.y, targetY, delta * 5)
        
        meshRef.current.material.emissiveIntensity = THREE.MathUtils.lerp(
            meshRef.current.material.emissiveIntensity,
            isActive ? 0.4 : 0,
            delta * 5
        )
        
        if (textRef.current) {
            textRef.current.fillOpacity = THREE.MathUtils.lerp(
                textRef.current.fillOpacity,
                isActive ? 1 : 0.05,
                delta * 5
            )
            textRef.current.position.y = meshRef.current.position.y + 0.6
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
                    roughness={0.2} 
                    metalness={0.1}
                />
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

export default function HouseScrollScene() {
    const groupRef = useRef()
    
    useEffect(() => {
        const handleScroll = () => {
            // Calculate progress excluding the first viewport height so it starts mostly when you begin scrolling down
            const totalScrollable = document.documentElement.scrollHeight - window.innerHeight
            const progress = totalScrollable > 0 ? window.scrollY / totalScrollable : 0
            if (groupRef.current && groupRef.current.parent) {
                groupRef.current.parent.userData.scrollProgress = Math.max(0, Math.min(1, progress))
            }
        }
        window.addEventListener('scroll', handleScroll, { passive: true })
        handleScroll()
        return () => window.removeEventListener('scroll', handleScroll)
    }, [])

    useFrame((state, delta) => {
        const offset = state.scene.userData.scrollProgress || 0
        
        if (groupRef.current) {
            // Spin and bring closer as we scroll
            groupRef.current.rotation.y = THREE.MathUtils.lerp(
                groupRef.current.rotation.y,
                offset * Math.PI * 2,
                delta * 3
            )
            
            groupRef.current.position.y = THREE.MathUtils.lerp(
                groupRef.current.position.y,
                -1 + offset * 1.5,
                delta * 3
            )
        }
    })

    const rooms = [
        { id: 'living', label: 'Living Room', pos: [0, 0, 2], size: [4, 0.5, 3], color: '#EFF6FF', scroll: 0.15 },
        { id: 'kitchen', label: 'Kitchen', pos: [-3, 0, -1], size: [3, 0.5, 3], color: '#F0FDF4', scroll: 0.35 },
        { id: 'master', label: 'Master Br.', pos: [2.5, 0, -1], size: [3.5, 0.5, 3], color: '#FFFBEB', scroll: 0.55 },
        { id: 'bath', label: 'Bathroom', pos: [-2, 0, -3.5], size: [2, 0.5, 2], color: '#F0FDFA', scroll: 0.75 },
        { id: 'bed2', label: 'Bedroom', pos: [2, 0, -3.5], size: [3, 0.5, 2], color: '#FFF7ED', scroll: 0.95 },
    ]

    return (
        <group ref={groupRef} position={[0, -1, 0]}>
            <ambientLight intensity={0.6} />
            <directionalLight position={[10, 20, 10]} intensity={1.5} castShadow shadow-mapSize={[2048, 2048]} />
            <Environment preset="city" />
            
            <mesh receiveShadow position={[0, -0.05, -0.5]}>
                <boxGeometry args={[11, 0.1, 10]} />
                <meshStandardMaterial color="#ffffff" roughness={1} />
            </mesh>
            
            {rooms.map(room => (
                <Room 
                    key={room.id}
                    position={room.pos}
                    size={room.size}
                    color={room.color}
                    label={room.label}
                    targetScroll={room.scroll}
                />
            ))}
            
            <ContactShadows position={[0, -0.1, 0]} opacity={0.6} scale={20} blur={2.5} far={4} />
        </group>
    )
}
