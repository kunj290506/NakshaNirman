/**
 * Image Analyzer
 * Analyzes uploaded floor plan images to extract plot boundaries
 */

/**
 * Analyze uploaded image to detect plot boundaries
 * @param {File} imageFile - Uploaded image file
 * @param {string} apiKey - Gemini API key for AI vision analysis
 * @returns {Promise<Object>} {success: boolean, boundary?: Array, dimensions?: Object, error?: string}
 */
export async function analyzeFloorPlanImage(imageFile, apiKey) {
    try {
        // Convert image to base64
        const base64Image = await fileToBase64(imageFile);
        
        // Use Gemini Vision API to analyze the image
        const analysisResult = await analyzeWithGeminiVision(base64Image, apiKey);
        
        return analysisResult;
    } catch (error) {
        console.error('Image analysis error:', error);
        return {
            success: false,
            error: error.message || 'Failed to analyze image'
        };
    }
}

/**
 * Convert image file to base64
 */
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            // Extract base64 data (remove data:image/xxx;base64, prefix)
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

/**
 * Analyze image using Gemini Vision API
 */
async function analyzeWithGeminiVision(base64Image, apiKey) {
    const GEMINI_VISION_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';
    
    const prompt = `You are an expert architectural CAD analyst. Analyze this floor plan image and extract the following information:

1. Plot/boundary dimensions (width and length in feet or meters)
2. Plot shape (rectangular, L-shaped, T-shaped, or irregular)
3. Any existing room layouts visible
4. Overall plot area if shown
5. Any dimension annotations visible in the image

Provide your response in JSON format:
{
  "plotDimensions": {
    "width": number,  // in feet
    "length": number, // in feet
    "unit": "feet" | "meters"
  },
  "plotShape": "rectangular" | "L-shaped" | "T-shaped" | "irregular",
  "totalAreaSqft": number,  // if visible
  "existingRooms": [
    {"type": "bedroom", "count": number},
    // ... other rooms if visible
  ],
  "boundary": [[x1,y1], [x2,y2], ...],  // corners in normalized coordinates (0-1 range)
  "notes": "Any additional observations"
}

If dimensions are not clearly visible, make reasonable estimates based on standard residential plot sizes.`;

    try {
        const response = await fetch(`${GEMINI_VISION_URL}?key=${apiKey}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contents: [{
                    parts: [
                        { text: prompt },
                        {
                            inline_data: {
                                mime_type: 'image/jpeg',
                                data: base64Image
                            }
                        }
                    ]
                }],
                generationConfig: {
                    temperature: 0.4,
                    maxOutputTokens: 2048
                }
            })
        });

        if (!response.ok) {
            throw new Error(`Gemini API error: ${response.statusText}`);
        }

        const data = await response.json();
        const aiResponse = data.candidates?.[0]?.content?.parts?.[0]?.text;

        if (!aiResponse) {
            throw new Error('No response from AI');
        }

        // Parse JSON from response
        const parsed = parseAIVisionResponse(aiResponse);
        
        return {
            success: true,
            ...parsed
        };

    } catch (error) {
        console.error('Gemini Vision API error:', error);
        return {
            success: false,
            error: error.message
        };
    }
}

/**
 * Parse AI vision response
 */
function parseAIVisionResponse(response) {
    // Try to extract JSON from response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    
    if (jsonMatch) {
        try {
            const parsed = JSON.parse(jsonMatch[0]);
            
            // Convert feet to mm if needed
            let widthMm, lengthMm;
            if (parsed.plotDimensions) {
                const unit = parsed.plotDimensions.unit || 'feet';
                const conversionFactor = unit === 'feet' ? 304.8 : 1000;
                
                widthMm = parsed.plotDimensions.width * conversionFactor;
                lengthMm = parsed.plotDimensions.length * conversionFactor;
            }
            
            // Normalize boundary coordinates to actual mm
            let boundaryMm = null;
            if (parsed.boundary && widthMm && lengthMm) {
                boundaryMm = parsed.boundary.map(([x, y]) => [
                    x * widthMm,
                    y * lengthMm
                ]);
            }
            
            return {
                plotDimensions: {
                    width: widthMm,
                    length: lengthMm,
                    widthFt: parsed.plotDimensions?.width,
                    lengthFt: parsed.plotDimensions?.length
                },
                plotShape: parsed.plotShape || 'rectangular',
                totalAreaSqft: parsed.totalAreaSqft,
                boundary: boundaryMm,
                existingRooms: parsed.existingRooms || [],
                notes: parsed.notes || ''
            };
        } catch (e) {
            console.warn('Failed to parse JSON from AI response:', e);
        }
    }
    
    // Fallback: return default rectangular plot
    return {
        plotDimensions: {
            width: 10972.8,  // 36 feet in mm
            length: 10972.8,
            widthFt: 36,
            lengthFt: 36
        },
        plotShape: 'rectangular',
        totalAreaSqft: 1296,
        boundary: null,
        existingRooms: [],
        notes: 'Could not extract precise dimensions from image. Using default 36x36 feet plot.'
    };
}

/**
 * Preview image before upload
 */
export function createImagePreview(file, callback) {
    const reader = new FileReader();
    reader.onload = (e) => {
        callback(e.target.result);
    };
    reader.readAsDataURL(file);
}

/**
 * Validate image file
 */
export function validateImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!validTypes.includes(file.type)) {
        return {
            valid: false,
            error: 'Please upload a valid image file (JPEG, PNG, or GIF)'
        };
    }
    
    if (file.size > maxSize) {
        return {
            valid: false,
            error: 'Image size must be less than 10MB'
        };
    }
    
    return { valid: true };
}
