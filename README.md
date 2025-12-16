### Hands-On Repository

The `falsification` folder contains the falsification example discussed during the lesson.

Before running the example, make sure to install all the required dependencies as described on the PSY-Taliro website, available [here](https://github.com/cpslab-asu/psy-taliro).

#### Overview: Teaching Scenario

This repository demonstrates **automated testing and validation** of cyber-physical systems through a realistic two-vehicle driving scenario:

**The Scenario:**
- A **lead vehicle** follows a predefined speed profile (simulating real-world traffic).
- An **ego (following) vehicle** is controlled by an Adaptive Cruise Control (ACC) system to maintain a safe distance from the lead vehicle.
- The key safety property: **the inter-vehicle distance must always remain positive** (no collision).

**What We Do:**

1. **Simulate the System** (scripts 0-1): We implement the vehicle dynamics, a simple driver model, and a rule-based ACC controller to understand how the system behaves.

2. **Falsify the Rule-Based ACC** (script 2): We use S-TalirO, an automated falsification tool, to search for scenarios (lead vehicle speed profiles) that violate the safety property. If found, we have a *falsification* — proof that the system can fail.

3. **Learn a Neural Network Controller** (script 3): We collect simulation data from many scenarios and train a neural network to replace the rule-based ACC controller. This demonstrates data-driven control.

4. **Deploy the Learned Controller** (script 4): We test the trained neural network controller in closed-loop simulation to observe its behavior.

5. **Falsify the Learned Controller** (script 5): We apply S-TalirO again to the neural network controller, searching for failure scenarios. This shows that learned controllers also need rigorous testing.

**Key Learning Goals for Students:**
- Understand the falsification-based testing approach for safety-critical systems.
- Compare rule-based vs. data-driven (neural network) control strategies.
- Recognize that both traditional and learned controllers require systematic validation.
- Gain hands-on experience with simulation, optimization, and formal verification tools.

---

The repository includes six Python files that implement the different steps of the falsification process (`#_name.py`), along with the `elements.py` file, which contains the relevant class implementations. Please refer to the comments at the top of these files for further details.
