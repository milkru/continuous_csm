#pragma once

class Camera
{
  public:
	Camera() = default;

	Camera(const glm::vec3& pos, const glm::vec3& target, const glm::vec3& up)
	    : cameraPosition(pos), cameraOrientation(glm::lookAt(pos, target, up)), upVector(up)
	{
	}

	void update(double deltaSeconds, const glm::vec2& mousePos, bool mousePressed)
	{
		if (mousePressed)
		{
			const glm::vec2 mouseDelta = mousePos - mousePosition;
			const glm::quat deltaRotation =
			    glm::quat(glm::vec3(-mouseSpeed * mouseDelta.y, mouseSpeed * mouseDelta.x, 0.0f));
			cameraOrientation = glm::normalize(deltaRotation * cameraOrientation);
			setUpVector(upVector);
		}

		mousePosition = mousePos;

		const glm::mat4 rotationMatrix = glm::mat4_cast(cameraOrientation);
		const glm::vec3 forward = -glm::vec3(rotationMatrix[0][2], rotationMatrix[1][2], rotationMatrix[2][2]);
		const glm::vec3 right = glm::vec3(rotationMatrix[0][0], rotationMatrix[1][0], rotationMatrix[2][0]);
		const glm::vec3 up = glm::cross(right, forward);

		glm::vec3 accelerationDir(0.0f);

		if (movement.forward)
		{
			accelerationDir += forward;
		}
		if (movement.backward)
		{
			accelerationDir -= forward;
		}
		if (movement.left)
		{
			accelerationDir -= right;
		}
		if (movement.right)
		{
			accelerationDir += right;
		}
		if (movement.up)
		{
			accelerationDir += up;
		}
		if (movement.down)
		{
			accelerationDir -= up;
		}

		if (movement.fastSpeed)
		{
			accelerationDir *= fastCoefficient;
		}

		if (accelerationDir == glm::vec3(0))
		{
			velocity -= velocity * std::min((1.0f / damping) * static_cast<float>(deltaSeconds), 1.0f);
		}
		else
		{
			velocity += accelerationDir * acceleration * static_cast<float>(deltaSeconds);
			const float speedLimit = movement.fastSpeed ? maxSpeed * fastCoefficient : maxSpeed;
			if (glm::length(velocity) > speedLimit)
			{
				velocity = glm::normalize(velocity) * speedLimit;
			}
		}

		cameraPosition += velocity * static_cast<float>(deltaSeconds);
	}

	glm::mat4 getViewMatrix() const
	{
		const glm::mat4 translation = glm::translate(glm::mat4(1.0f), -cameraPosition);
		const glm::mat4 rotation = glm::mat4_cast(cameraOrientation);
		return rotation * translation;
	}

	void setUpVector(const glm::vec3& up)
	{
		const glm::mat4 view = getViewMatrix();
		const glm::vec3 direction = -glm::vec3(view[0][2], view[1][2], view[2][2]);
		cameraOrientation = glm::lookAt(cameraPosition, cameraPosition + direction, up);
	}

  public:
	struct Movement
	{
		bool forward = false;
		bool backward = false;
		bool left = false;
		bool right = false;
		bool up = false;
		bool down = false;
		bool fastSpeed = false;
	} movement;

	float mouseSpeed = 4.0f;
	float acceleration = 150.0f;
	float damping = 0.2f;
	float maxSpeed = 750.0f;
	float fastCoefficient = 10.0f;

  private:
	glm::vec2 mousePosition = glm::vec2(0);
	glm::vec3 cameraPosition = glm::vec3(0.0f, 10.0f, 10.0f);
	glm::quat cameraOrientation = glm::quat(glm::vec3(0));
	glm::vec3 velocity = glm::vec3(0.0f);
	glm::vec3 upVector = glm::vec3(0.0f, 0.0f, 1.0f);
};
