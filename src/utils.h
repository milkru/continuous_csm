#pragma once

// ---- RAII scope guard -------------------------------------------------------

template <typename F> struct ScopeExit_
{
	F fn_;
	~ScopeExit_() { fn_(); }
};

struct ScopeExitHelper_ {};

template <typename F> ScopeExit_<F> operator+(ScopeExitHelper_, F&& fn)
{
	return { std::forward<F>(fn) };
}

#define SCOPE_EXIT auto _se_##__LINE__ = ScopeExitHelper_{} + [&]() noexcept

// ---- Normal / snorm packing -------------------------------------------------

inline glm::vec2 msign(glm::vec2 v)
{
	return glm::vec2(v.x >= 0.0f ? 1.0f : -1.0f, v.y >= 0.0f ? 1.0f : -1.0f);
}

inline uint16_t packOctahedral16(glm::vec3 n)
{
	n /= (glm::abs(n.x) + glm::abs(n.y) + glm::abs(n.z));
	const glm::vec2 v = (n.z >= 0.0f) ? glm::vec2(n.x, n.y)
	                                   : (glm::vec2(1.0f) - glm::abs(glm::vec2(n.y, n.x))) * msign(glm::vec2(n));
	const glm::uvec2 d = glm::uvec2(glm::round(127.5f + v * 127.5f));
	return d.x | (d.y << 8u);
}
