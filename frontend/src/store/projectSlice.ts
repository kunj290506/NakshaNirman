import { createSlice, PayloadAction } from '@reduxjs/toolkit'

interface Room {
    name: string
    type: string
    area: number
}

interface ProjectState {
    jobId: string | null
    status: 'idle' | 'uploading' | 'processing' | 'completed' | 'failed'
    progress: number
    stage: string
    message: string
    requirements: {
        bedrooms: number
        bathrooms: number
        style: string
        features: string[]
        budget: string
    }
    results: {
        dxfUrl: string | null
        pdfUrl: string | null
        gltfUrl: string | null
        videoUrl: string | null
        rooms: Room[]
        totalArea: number
    } | null
}

const initialState: ProjectState = {
    jobId: null,
    status: 'idle',
    progress: 0,
    stage: '',
    message: '',
    requirements: {
        bedrooms: 3,
        bathrooms: 2,
        style: 'modern',
        features: [],
        budget: 'standard',
    },
    results: null,
}

const projectSlice = createSlice({
    name: 'project',
    initialState,
    reducers: {
        setJobId: (state, action: PayloadAction<string>) => {
            state.jobId = action.payload
        },
        setStatus: (state, action: PayloadAction<ProjectState['status']>) => {
            state.status = action.payload
        },
        setProgress: (state, action: PayloadAction<{ progress: number; stage: string; message: string }>) => {
            state.progress = action.payload.progress
            state.stage = action.payload.stage
            state.message = action.payload.message
        },
        setRequirements: (state, action: PayloadAction<Partial<ProjectState['requirements']>>) => {
            state.requirements = { ...state.requirements, ...action.payload }
        },
        setResults: (state, action: PayloadAction<ProjectState['results']>) => {
            state.results = action.payload
            state.status = 'completed'
        },
        resetProject: () => initialState,
    },
})

export const { setJobId, setStatus, setProgress, setRequirements, setResults, resetProject } = projectSlice.actions
export default projectSlice.reducer
