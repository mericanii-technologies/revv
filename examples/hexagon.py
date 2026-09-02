import pygame
import math
import sys

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Ball Bouncing in Spinning Hexagon")
    clock = pygame.time.Clock()

    # Constants
    WIDTH, HEIGHT = 800, 600
    CENTER = (WIDTH // 2, HEIGHT // 2)
    HEX_RADIUS = 250
    BALL_RADIUS = 15
    GRAVITY = 0.5
    FRICTION = 0.99
    BOUNCE_DAMPING = 0.8
    HEX_ANGULAR_VELOCITY = 0.01  # radians per frame

    # Ball state
    ball_x, ball_y = CENTER[0], CENTER[1] - 100
    ball_vx, ball_vy = 2.0, 0.0

    # Hexagon state
    hex_angle = 0.0

    # Precompute hexagon vertices (unit circle, scaled later)
    def get_hex_vertices(center_x, center_y, radius, angle):
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def closest_point_on_segment(px, py, ax, ay, bx, by):
        """Find the closest point on segment AB to point P."""
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return ax, ay
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        cx = ax + t * dx
        cy = ay + t * dy
        return cx, cy

    def point_to_segment_distance(px, py, ax, ay, bx, by):
        """Return distance from point P to segment AB, and the closest point."""
        cx, cy = closest_point_on_segment(px, py, ax, ay, bx, by)
        dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        return dist, (cx, cy)

    def reflect_velocity(vx, vy, nx, ny, damping):
        """Reflect velocity vector off a surface with normal (nx, ny)."""
        dot = vx * nx + vy * ny
        rvx = (vx - 2 * dot * nx) * damping
        rvy = (vy - 2 * dot * ny) * damping
        return rvx, rvy

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Update hexagon angle
        hex_angle += HEX_ANGULAR_VELOCITY

        # Apply gravity
        ball_vy += GRAVITY

        # Apply friction
        ball_vx *= FRICTION
        ball_vy *= FRICTION

        # Update ball position
        ball_x += ball_vx
        ball_y += ball_vy

        # Get current hexagon vertices
        vertices = get_hex_vertices(CENTER[0], CENTER[1], HEX_RADIUS, hex_angle)

        # Check collision with each edge
        for i in range(6):
            ax, ay = vertices[i]
            bx, by = vertices[(i + 1) % 6]

            # Compute edge normal (pointing inward)
            ex = bx - ax
            ey = by - ay
            # Normal perpendicular to edge
            nx = -ey
            ny = ex
            # Normalize
            nlen = math.sqrt(nx * nx + ny * ny)
            if nlen > 0:
                nx /= nlen
                ny /= nlen

            # Check if normal points inward (toward center)
            mid_x = (ax + bx) / 2
            mid_y = (ay + by) / 2
            to_center_x = CENTER[0] - mid_x
            to_center_y = CENTER[1] - mid_y
            if nx * to_center_x + ny * to_center_y < 0:
                nx = -nx
                ny = -ny

            # Distance from ball center to edge segment
            dist, closest = point_to_segment_distance(ball_x, ball_y, ax, ay, bx, by)

            if dist < BALL_RADIUS:
                # Collision detected
                # Push ball out of the wall
                overlap = BALL_RADIUS - dist
                if overlap > 0:
                    ball_x += nx * overlap
                    ball_y += ny * overlap

                # Compute velocity of the wall at the contact point
                # The wall is rotating about CENTER with angular velocity HEX_ANGULAR_VELOCITY
                # Velocity of a point at position (px, py) due to rotation:
                # v = omega x r, where r is vector from center to point
                # In 2D: v_x = -omega * (py - cy), v_y = omega * (px - cx)
                contact_x, contact_y = closest
                wall_vx = -HEX_ANGULAR_VELOCITY * (contact_y - CENTER[1])
                wall_vy = HEX_ANGULAR_VELOCITY * (contact_x - CENTER[0])

                # Relative velocity of ball with respect to wall
                rel_vx = ball_vx - wall_vx
                rel_vy = ball_vy - wall_vy

                # Check if ball is moving toward the wall
                rel_dot_n = rel_vx * nx + rel_vy * ny
                if rel_dot_n < 0:
                    # Reflect relative velocity
                    rvx, rvy = reflect_velocity(rel_vx, rel_vy, nx, ny, BOUNCE_DAMPING)

                    # Convert back to absolute velocity
                    ball_vx = rvx + wall_vx
                    ball_vy = rvy + wall_vy

        # Draw
        screen.fill((30, 30, 40))

        # Draw hexagon
        hex_points = [(int(x), int(y)) for x, y in vertices]
        pygame.draw.polygon(screen, (100, 150, 255), hex_points, 3)

        # Draw ball
        pygame.draw.circle(screen, (255, 100, 100), (int(ball_x), int(ball_y)), BALL_RADIUS)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
