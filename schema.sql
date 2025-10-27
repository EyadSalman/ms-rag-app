-- ===============================================
-- 🧠 DetectMS Database Schema
-- ===============================================

-- 1️⃣ Users Table
CREATE TABLE IF NOT EXISTS public.users (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  role TEXT CHECK (role IN ('admin', 'user')) DEFAULT 'user',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 2️⃣ MRI Images Table
CREATE TABLE IF NOT EXISTS public.mri_images (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid REFERENCES public.users(id) ON DELETE CASCADE,
  image_url TEXT,
  uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 3️⃣ Chat History Table
CREATE TABLE IF NOT EXISTS public.chat_history (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id uuid REFERENCES public.users(id) ON DELETE CASCADE,
  query TEXT,
  response TEXT,
  agent_type TEXT CHECK (agent_type IN ('mri', 'research')) DEFAULT 'research',
  sources JSONB,  -- store retrieved documents / URLs
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 4️⃣ Research Sources Table
CREATE TABLE IF NOT EXISTS public.research_sources (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  title TEXT,
  authors TEXT,
  year INT,
  source_url TEXT,
  pdf_path TEXT,
  embedding_model TEXT,
  chunk_count INT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- 5️⃣ MRI Results Table
CREATE TABLE IF NOT EXISTS public.mri_results (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT,
  diagnosis TEXT,
  confidence FLOAT,
  image_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);
