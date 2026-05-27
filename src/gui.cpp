#include <cfloat>
#include <cstdio>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include <lvk/HelpersImGui.h>
#include <lvk/LVK.h>

#include "gui.h"

extern std::unique_ptr<lvk::IContext> ctx_;
extern int32_t width_;
extern int32_t height_;
extern uint64_t pipelineTimestamps[];
extern double timestampBeginRendering;
extern double timestampEndRendering;

void showTimeGPU()
{
	const double toMS = ctx_->getTimestampPeriodToMs();
	auto getTimespan = [toMS](GPUTimestamp begin) -> double
	{ return double(pipelineTimestamps[begin + 1] - pipelineTimestamps[begin]) * toMS; };

	struct sTimeStats
	{
		enum
		{
			kNumTimelines = 4
		};

		struct MinMax
		{
			float vmin = FLT_MAX;
			float vmax = 0.0f;
		};

		double add(uint32_t pass, const char* name, double value)
		{
			LVK_ASSERT(pass < kNumTimelines);
			names[pass] = name;
			const float prev = timelines[pass].empty() ? (float)value : timelines[pass].back();
			timelines[pass].push_back(0.9f * prev + 0.1f * (float)value);
			if (timelines[pass].size() > 128)
			{
				timelines[pass].erase(timelines[pass].begin());
			}
			avg[pass] = (float)value;
			return value;
		}

		void updateMinMax()
		{
			for (uint32_t i = 0; i != kNumTimelines; i++)
			{
				float minT = FLT_MAX;
				float maxT = 0.0f;
				for (float v : timelines[i])
				{
					if (v < minT)
					{
						minT = v;
					}
					if (v > maxT)
					{
						maxT = v;
					}
				}
				minmax[i] = { minT, maxT };
			}
		}

		std::vector<float> timelines[kNumTimelines] = {};
		MinMax minmax[kNumTimelines] = {};
		float avg[kNumTimelines] = {};
		const char* names[kNumTimelines] = {};
		const glm::vec4 colors[kNumTimelines] = { glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
			                                      glm::vec4(0.749f, 0.847f, 0.847f, 1.0f),
			                                      glm::vec4(1.0f, 0.0f, 0.0f, 1.0f) };
	};

	static sTimeStats stats;

	const double timeScene = stats.add(1, " Scene", getTimespan(GPUTimestamp_BeginSceneRendering));
	const double timePresent = stats.add(2, " Present", getTimespan(GPUTimestamp_BeginPresent));
	const double timeGPU = timeScene + timePresent;
	stats.add(0, "GPU", timeGPU);
	stats.add(3, "CPU", (timestampEndRendering - timestampBeginRendering) * 1000);
	stats.updateMinMax();

	char text[128];
	snprintf(text, sizeof(text), "GPU: %6.02f ms   (Scene: %.02f   Present: %.02f)", timeGPU, timeScene, timePresent);

	const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
	                               ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
	                               ImGuiWindowFlags_NoNav;
	ImGui::SetNextWindowBgAlpha(0.8f);
	ImGui::SetNextWindowPos({ 20.0f, (float)height_ * 0.8f }, ImGuiCond_Appearing);
	ImGui::SetNextWindowSize({ (float)width_ * 0.4f, 0 });
	ImGui::Begin("GPU Stats", nullptr, flags);
	ImGui::Text("%s", text);

	auto Sparkline = [](const char* id, const float* values, int32_t count, float min_v, float max_v, const ImVec4& col,
	                    const ImVec2& size)
	{
		const ImVec2 pos = ImGui::GetCursorScreenPos();
		const float w = size.x < 0.0f ? ImGui::GetContentRegionAvail().x : size.x;
		const ImVec2 sz = ImVec2(w, size.y);
		ImGui::InvisibleButton(id, sz);

		if (count < 2 || min_v >= max_v)
		{
			return;
		}

		ImDrawList* dl = ImGui::GetWindowDrawList();
		const float range = max_v - min_v;
		const float baseY = pos.y + sz.y;
		const ImU32 lineCol = ImGui::ColorConvertFloat4ToU32(col);
		const ImU32 fillCol = ImGui::ColorConvertFloat4ToU32(ImVec4(col.x, col.y, col.z, col.w * 0.25f));

		auto xAt = [&](int32_t i) { return pos.x + (float)i / (float)(count - 1) * sz.x; };
		auto yAt = [&](float v) { return baseY - (v - min_v) / range * sz.y; };

		for (int32_t i = 0; i < count - 1; i++)
		{
			const float x0 = xAt(i), y0 = yAt(values[i]);
			const float x1 = xAt(i + 1), y1 = yAt(values[i + 1]);
			dl->AddQuadFilled(ImVec2(x0, baseY), ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(x1, baseY), fillCol);
			dl->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), lineCol, 1.5f);
		}
	};

	auto RowLeadIn = [](const char* stage, float value, const ImVec4& color)
	{
		ImGui::TableSetColumnIndex(0);
		ImGui::TextColored(color, "%s", stage);
		ImGui::TableSetColumnIndex(1);
		ImGui::TextColored(color, "%6.02f", value);
		ImGui::TableSetColumnIndex(2);
	};

	if (ImGui::BeginTable("##table", 3, ImGuiTableFlags_None, ImVec2(-1, 0)))
	{
		const ImGuiTableColumnFlags colFlags = ImGuiTableColumnFlags_NoSort;
		ImGui::TableSetupColumn("Stage", colFlags);
		ImGui::TableSetupColumn("Time (ms)", colFlags);
		ImGui::TableSetupColumn("Graph", colFlags | ImGuiTableColumnFlags_WidthStretch);
		ImGui::TableHeadersRow();
		for (uint32_t i = 0; i != sTimeStats::kNumTimelines; i++)
		{
			ImGui::TableNextRow();
			const glm::vec4& colorVec = stats.colors[i];
			const ImVec4 color(colorVec.x, colorVec.y, colorVec.z, colorVec.w);
			RowLeadIn(stats.names[i], stats.avg[i], color);
			if (stats.avg[i] > 0.01f)
			{
				ImGui::PushID((int32_t)i);
				Sparkline("##spark", stats.timelines[i].data(), (int32_t)stats.timelines[i].size(),
				          stats.minmax[i].vmin * 0.8f, stats.minmax[i].vmax * 1.2f, color, ImVec2(-1, 30));
				ImGui::PopID();
			}
		}
		ImGui::EndTable();
	}
	ImGui::End();
}
