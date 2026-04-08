#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>

/**
 * A keypoint representing a specific point on a hand
 */
struct Keypoint {
    float x;
    float y;
    float confidence;
};

/**
 * Represents the side of the hand (left, right, or unknown) with utility functions for string conversion and comparisons
 * Also supports optional implicit conversion to enum value for switch statements if HANDSIDE_ENABLE_SWITCH is set
 */
class HandSide {
    public:
        enum Value {
            LEFT,
            RIGHT,
            UNKNOWN
        };
        
        HandSide() = default;
        constexpr HandSide(Value v) : value(v) {}

        std::string toString() const {
            switch (value) {
                case LEFT: return "LEFT";
                case RIGHT: return "RIGHT";
                case UNKNOWN: return "UNKNOWN";
                default: return "INVALID";
            }
        }

        constexpr bool operator==(HandSide h) const { return value == h.value; }
        constexpr bool operator!=(HandSide h) const { return value != h.value; }

        Value getValue() const { return value; }
    private:
        Value value = UNKNOWN;
};

/**
 * A detected hand with bounding box, confidence, side, and all of its keypoints
 */
struct Hand {
    float x;
    float y;
    float width;
    float height;
    float confidence;
    HandSide side;
    std::vector<Keypoint> keypoints;
    // Keypoints are:
        // 0: Wrist
        // 1-4: Thumb (CMC, MCP, IP, Tip)
        // 5-8: Index (MCP, PIP, DIP, Tip)
        // 9-12: Middle (MCP, PIP, DIP, Tip)
        // 13-16: Ring (MCP, PIP, DIP, Tip)
        // 17-20: Pinky (MCP, PIP, DIP, Tip

};
    
#endif // TYPES_H