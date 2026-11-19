import { serve } from "./serve.js";
import { openStore } from "./store.js";

serve(self, (location) => openStore(location));
