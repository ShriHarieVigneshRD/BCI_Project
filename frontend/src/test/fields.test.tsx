import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { NumberField, TextField } from "../components/ui/Field";
import { Modal } from "../components/ui/Misc";
import { baselineInside } from "../pages/PreprocessingPage";

function NameInModal() {
  const [v, setV] = useState("");
  // inline onClose, as every caller writes it: a new function on each render
  return <Modal open onClose={() => undefined} title="Dialog"><TextField label="Name" value={v} onChange={setV} /></Modal>;
}

describe("text input in a modal", () => {
  it("keeps focus while typing", async () => {
    render(<NameInModal />);
    const input = screen.getByLabelText("Name");
    await userEvent.setup().type(input, "Lab study 2");
    expect(input).toHaveFocus();
    expect(input).toHaveValue("Lab study 2");
  });
});

function Baseline() {
  const [v, setV] = useState<number | null>(0);
  return <><NumberField label="From (s)" value={v} onChange={setV} /><output>{String(v)}</output></>;
}

describe("number input", () => {
  it("accepts a negative value typed from an empty field", async () => {
    render(<Baseline />);
    const input = screen.getByLabelText("From (s)");
    const user = userEvent.setup();
    await user.clear(input);
    await user.type(input, "-0.2");
    expect(screen.getByRole("status")).toHaveTextContent("-0.2");
    expect(input).toHaveFocus();
  });
});

describe("baseline window", () => {
  it("is moved inside the epoch when enabled", () => {
    expect(baselineInside({ tmin: -0.2, tmax: 0 }, [0, 2])).toEqual({ tmin: 0, tmax: 0.2 }); // sliding windows
    expect(baselineInside({ tmin: -0.2, tmax: 0 }, [-0.5, 1])).toEqual({ tmin: -0.2, tmax: 0 }); // already valid
    expect(baselineInside({ tmin: -1, tmax: 0 }, [-0.5, 1])).toEqual({ tmin: -0.5, tmax: 0 });
    expect(baselineInside({ tmin: -0.2, tmax: 0 }, [0, 0.25])).toEqual({ tmin: 0, tmax: 0.125 });
  });
});
