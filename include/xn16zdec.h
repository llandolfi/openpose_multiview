#pragma once

// comment if openni is defined
typedef int XnStatus;
typedef int XnBool;
typedef unsigned char XnUInt8;
typedef signed char XnInt8;
typedef unsigned int XnUInt32;
typedef unsigned short XnUInt16;
typedef signed short XnInt16;

extern "C"
{
XnStatus XnStreamUncompressDepth16Z(const XnUInt8* pInput,
                                    const XnUInt32 nInputSize,
                                    XnUInt16* pOutput,
                                    XnUInt32* pnOutputSize);
XnStatus XnStreamUncompressDepth16ZWithEmbTable(const XnUInt8* pInput,
                                                const XnUInt32 nInputSize,
                                                XnUInt16* pOutput,
                                                XnUInt32* pnOutputSize);
XnStatus XnStreamCompressDepth16Z(const XnUInt16* pInput,
                                  const XnUInt32 nInputSize,
                                  XnUInt8* pOutput,
                                  XnUInt32* pnOutputSize);
XnStatus XnStreamCompressDepth16ZWithEmbTable(const XnUInt16* pInput,
                                              const XnUInt32 nInputSize,
                                              XnUInt8* pOutput,
                                              XnUInt32* pnOutputSize,
                                              XnUInt16 nMaxValue);
XnStatus XnStreamUncompressImage8Z(const XnUInt8* pInput,
                                   const XnUInt32 nInputSize,
                                   XnUInt8* pOutput,
                                   XnUInt32* pnOutputSize);
XnStatus XnStreamCompressImage8Z(const XnUInt8* pInput,
                                 const XnUInt32 nInputSize,
                                 XnUInt8* pOutput,
                                 XnUInt32* pnOutputSize);
}