import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, Navigate, RouterProvider, useRouteError, type RouteObject } from "react-router";
import { AppShell } from "./components/layout/AppShell";
import { DatasetSplitsPage } from "./pages/DatasetSplitsPage";
import { PreprocessingPage } from "./pages/PreprocessingPage";
import { ArchitectureExplorerPage } from "./pages/architecture/ArchitectureExplorerPage";
import { BattleArenaPage } from "./pages/BattleArenaPage";
import { TensorDetectivePage } from "./pages/TensorDetectivePage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { EmptyState } from "./components/ui/Misc";

function RouteError() {
  const err = useRouteError() as Error | undefined;
  return (
    <div className="mx-auto max-w-3xl p-10">
      <EmptyState title="This view failed to render">{err?.message ?? "Unknown error"}</EmptyState>
    </div>
  );
}

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Navigate to="/dataset" replace /> },
      { path: "dataset", element: <DatasetSplitsPage /> },
      { path: "preprocessing", element: <PreprocessingPage /> },
      { path: "architecture", element: <ArchitectureExplorerPage /> },
      { path: "architecture/:tab", element: <ArchitectureExplorerPage /> },
      { path: "arena", element: <BattleArenaPage /> },
      { path: "tensors", element: <TensorDetectivePage /> },
      { path: "experiments", element: <ExperimentsPage /> },
      { path: "*", element: <EmptyState title="Page not found">Use the navigation above.</EmptyState> },
    ],
  },
];

export function makeQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { refetchOnWindowFocus: false, staleTime: 5_000 } },
  });
}

const queryClient = makeQueryClient();
const router = createBrowserRouter(routes);

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
