#pragma once

enum GPUTimestamp
{
	GPUTimestamp_BeginSceneRendering = 0,
	GPUTimestamp_EndSceneRendering,
	GPUTimestamp_BeginPresent,
	GPUTimestamp_EndPresent,
	GPUTimestamp_NUM_TIMESTAMPS
};

void showTimeGPU();
