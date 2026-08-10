--[[
  Photo-Derush Darktable Lua Plugin
  Integrates photo-derush ML active learning & burst grouping into Lighttable
--]]

local dt = require "darktable"

local update_panel_stats = nil

local function get_target_images()
    local act_imgs = nil
    pcall(function() act_imgs = dt.gui.action_images end)
    if act_imgs and #act_imgs > 0 then
        return act_imgs
    end
    local hover_img = nil
    pcall(function() hover_img = dt.gui.hover() end)
    if not hover_img then
        pcall(function() hover_img = dt.gui.hovered end)
    end
    if hover_img then
        return { hover_img }
    end
    local sel = nil
    pcall(function() sel = dt.gui.selection() end)
    if sel and #sel > 0 then
        return sel
    end
    return {}
end

-- Helper to get common root directory for all images in active collection
local function get_collection_root_dir(images)
    if not images or #images == 0 then return os.getenv("USERPROFILE") end
    local common_dir = nil
    for _, img in ipairs(images) do
        local pth = img.path
        if pth and pth ~= "" then
            if not common_dir then
                common_dir = pth
            else
                while common_dir ~= "" and not pth:lower():find(common_dir:lower(), 1, true) do
                    common_dir = common_dir:match("^(.+)[/\\][^/\\]+$") or ""
                end
            end
        end
    end
    if not common_dir or common_dir == "" then
        local first = images[1]
        common_dir = (first and first.film and first.film.path) or (first and first.path)
    end
    if not common_dir or common_dir == "" then
        common_dir = os.getenv("USERPROFILE")
    end
    return common_dir
end

-- Exclusive Keep / Trash Shortcuts (K / T hotkeys)
dt.register_event("derush_set_keep", "shortcut", function(event, shortcut)
    local ok, err = pcall(function()
        local images = get_target_images()
        if #images > 0 then
            for _, img in ipairs(images) do
                img.red = false
                img.green = true
                img.rating = 5
            end
            if update_panel_stats then pcall(update_panel_stats) end
            dt.print(string.format("Derush: Marked %d photo(s) as Keep (Green + 5 Stars)", #images))
        else
            dt.print("Derush: Select or hover over a photo to mark as Keep")
        end
    end)
    if not ok then
        log_debug("SHORTCUT K ERROR: " .. tostring(err))
        dt.print("Derush Error: " .. tostring(err))
    end
end, "Derush: Set Exclusive Keep (Green + 5 Stars)")

dt.register_event("derush_set_trash", "shortcut", function(event, shortcut)
    local ok, err = pcall(function()
        local images = get_target_images()
        if #images > 0 then
            for _, img in ipairs(images) do
                img.green = false
                img.red = true
                img.rating = -1
            end
            if update_panel_stats then pcall(update_panel_stats) end
            dt.print(string.format("Derush: Marked %d photo(s) as Trash (Red + Rejected)", #images))
        else
            dt.print("Derush: Select or hover over a photo to mark as Trash")
        end
    end)
    if not ok then
        log_debug("SHORTCUT T ERROR: " .. tostring(err))
        dt.print("Derush Error: " .. tostring(err))
    end
end, "Derush: Set Exclusive Trash (Red + Rejected)")

-- Hot-Reload helper function
local function hot_reload()
    local plugin_path = os.getenv("LOCALAPPDATA") .. "/darktable/lua/derush/derush.lua"
    package.loaded["derush/derush"] = nil
    dt.print("Hot-reloading Derush plugin from disk...")
    local ok, err = pcall(dofile, plugin_path)
    if not ok then
        dt.print("Hot-reload error: " .. tostring(err))
    else
        dt.print("Derush plugin hot-reloaded successfully!")
    end
end

-- Log to debug file
local function log_debug(msg)
    local log_path = os.getenv("LOCALAPPDATA") .. [[\darktable\derush_debug.log]]
    local f = io.open(log_path, "a")
    if f then
        f:write("[" .. os.date("%Y-%m-%d %H:%M:%S") .. "] " .. msg .. "\n")
        f:close()
    end
end

local function run_derush_command(cmd_name, folder_path, labels_json, files_json)
    local is_windows = (package.config:sub(1, 1) == "\\")

    local python_bin = os.getenv("USERPROFILE") .. "\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\photo-app-rBz6-pE0-py3.12\\Scripts\\python.exe"
    local script_path = os.getenv("LOCALAPPDATA") .. "\\darktable\\lua\\derush\\cli_bridge.py"
    local temp_dir_path = os.getenv("LOCALAPPDATA") .. "\\darktable\\temp_directory.txt"
    local temp_labels_path = os.getenv("LOCALAPPDATA") .. "\\darktable\\temp_labels.json"
    local temp_files_path = os.getenv("LOCALAPPDATA") .. "\\darktable\\temp_files.json"

    if folder_path and folder_path ~= "" then
        local f = io.open(temp_dir_path, "w")
        if f then
            f:write(folder_path)
            f:close()
        end
    end

    local extra_arg = ""
    if labels_json and labels_json ~= "" then
        local lf = io.open(temp_labels_path, "w")
        if lf then
            lf:write(labels_json)
            lf:close()
            extra_arg = extra_arg .. string.format(' --labels-file "%s"', temp_labels_path)
        end
    end

    if files_json and files_json ~= "" then
        local ff = io.open(temp_files_path, "w")
        if ff then
            ff:write(files_json)
            ff:close()
            extra_arg = extra_arg .. string.format(' --files-file "%s"', temp_files_path)
        end
    end

    local inner_cmd = string.format('"%s" "%s" %s --directory-file "%s"%s',
        python_bin, script_path, cmd_name, temp_dir_path, extra_arg)

    local command = inner_cmd
    if is_windows then
        -- Wrap with a single outer pair of quotes so Windows cmd.exe /c strips outer quotes cleanly
        command = string.format('"%s"', inner_cmd)
    end

    log_debug("COMMAND: " .. command)

    local handle = io.popen(command .. " 2>&1")
    if not handle then
        dt.print("Derush Error: Could not execute command handle")
        log_debug("ERROR: io.popen returned nil")
        return nil
    end

    local result = handle:read("*a")
    handle:close()

    log_debug("OUTPUT:\n" .. tostring(result))

    return result
end

-- Cache for created tag handles to prevent redundant SQLite queries
local created_tags_cache = {}

-- Function to attach Derush ML Score tag to image in Darktable (optimized for speed)
local function set_image_derush_score(img, score)
    local tag_name = string.format("derush|score_%0.2f", score)

    -- 1. Reuse cached tag handle or create once
    local score_tag = created_tags_cache[tag_name]
    if not score_tag then
        score_tag = dt.tags.create(tag_name)
        created_tags_cache[tag_name] = score_tag
    end

    -- 2. Detach outdated score tags if present
    local existing_tags = dt.tags.get_tags(img)
    local already_attached = false
    if existing_tags then
        for _, t in ipairs(existing_tags) do
            if t.name == tag_name then
                already_attached = true
            elseif t.name:find("^derush|score_") then
                dt.tags.detach(t, img)
            end
        end
    end

    -- 3. Attach score tag if not already attached
    if not already_attached and score_tag then
        dt.tags.attach(score_tag, img)
    end
end

-- Panel Status Labels
local label_stats_selected       = dt.new_widget("label") { label = "Photos in View: -" }
local label_stats_manual         = dt.new_widget("label") { label = "Manual Labels: -" }
local label_stats_predictions    = dt.new_widget("label") { label = "Auto Predicted: -" }
local label_stats_trained        = dt.new_widget("label") { label = "Training Data: -" }
local label_stats_score          = dt.new_widget("label") { label = "Model Accuracy: -" }
local label_stats_scores_detail  = dt.new_widget("label") { label = "Analysis Detail: -" }

-- UI Panel Buttons
local predict_btn = dt.new_widget("button") {
    label = "✨ Predict / Score Photos",
    tooltip = "Compute ML prediction scores for photos in active collection",
    clicked_callback = function(widget)
        local ok, err = pcall(function()
            local images = dt.gui.selection()
            if not images or #images == 0 then
                images = {}
                local col_ok, col = pcall(function() return dt.collection end)
                if col_ok and col then
                    for i = 1, #col do
                        table.insert(images, col[i])
                    end
                else
                    for i = 1, #dt.database do
                        table.insert(images, dt.database[i])
                    end
                end
            end

            local total_count = #images
            if total_count == 0 then
                dt.print("Derush Error: No images found to predict")
                return
            end

            label_stats_selected.label = string.format("Selected Photos: %d", total_count)

            local job = nil
            pcall(function()
                job = dt.gui.create_job("Derush: Scoring " .. total_count .. " photos...")
            end)

            local folder_path = get_collection_root_dir(images)

            local file_paths = {}
            for _, img in ipairs(images) do
                local pth = img.path
                local fn = img.filename
                local full_p = (pth and pth ~= "") and (pth .. "/" .. fn) or fn
                table.insert(file_paths, string.format('"%s"', full_p:gsub("\\", "/")))
            end
            local files_json = "[" .. table.concat(file_paths, ",") .. "]"

            dt.print(string.format("Derush: Running ML predictions for %d images...", total_count))
            log_debug("SCORING: folder=" .. tostring(folder_path) .. " images=" .. total_count)

            if job then pcall(function() job.percent = 0.10 end) end
            local raw_json = run_derush_command("predict", folder_path, nil, files_json)

            if not raw_json or raw_json == "" then
                if job then pcall(function() job.valid = false end) end
                dt.print("Derush Error: Backend returned empty response")
                return
            end

            local predictions = {}
            for filename, score in raw_json:gmatch('"([^"]+)":%s*([%d%.]+)') do
                local val = tonumber(score)
                if val then
                    predictions[filename] = val
                    predictions[filename:lower()] = val
                end
            end

            local count = 0
            local matched_count = 0
            local count_keep = 0
            local count_trash = 0
            local sum_keep = 0
            local sum_trash = 0
            local sum_total = 0

            for i, img in ipairs(images) do
                local fn = img.filename or ""
                local pth = img.path or ""
                local fn_stem = fn:match("^(.+)%..+$") or fn

                local score = predictions[fn]
                    or predictions[fn:lower()]
                    or predictions[fn_stem]
                    or predictions[fn_stem:lower()]
                    or predictions[pth]
                    or predictions[pth:lower()]

                if not score then
                    for k, v in pairs(predictions) do
                        local k_stem = k:match("^(.+)%..+$") or k
                        if k_stem:lower() == fn_stem:lower() then
                            score = v
                            break
                        end
                    end
                end

                if score then
                    matched_count = matched_count + 1
                else
                    score = 0.50
                end

                if score >= 0.50 then
                    count_keep = count_keep + 1
                    sum_keep = sum_keep + score
                else
                    count_trash = count_trash + 1
                    sum_trash = sum_trash + score
                end
                sum_total = sum_total + score

                set_image_derush_score(img, score)
                count = count + 1
                if count % 100 == 0 or count == total_count then
                    dt.print(string.format("Derush: Tagging scores... %d/%d photos done", count, total_count))
                end
                if job then
                    pcall(function() job.percent = 0.10 + 0.90 * (count / total_count) end)
                end
            end

            if job then pcall(function() job.valid = false end) end

            local avg_total = count > 0 and (sum_total / count) or 0
            local avg_keep  = count_keep > 0 and (sum_keep / count_keep) or 0
            local avg_trash = count_trash > 0 and (sum_trash / count_trash) or 0

            label_stats_selected.label      = string.format("Analysed: %d (%d matched)", count, matched_count)
            label_stats_predictions.label   = string.format("Predicted: %d Keep / %d Trash", count_keep, count_trash)
            label_stats_scores_detail.label = string.format("Avg Scores: Keep %.2f | Trash %.2f", avg_keep, avg_trash)

            log_debug(string.format("SCORING COMPLETE: Matched %d/%d. Keep: %d (avg %.2f), Trash: %d (avg %.2f)",
                matched_count, count, count_keep, avg_keep, count_trash, avg_trash))
            dt.print(string.format("Derush: Analysed %d photos! %d Keep (avg %.2f), %d Trash (avg %.2f)",
                count, count_keep, avg_keep, count_trash, avg_trash))
        end)
        if not ok then
            log_debug("SCORING ERROR: " .. tostring(err))
            dt.print("Derush Error: " .. tostring(err))
        end
    end
}

local train_btn = dt.new_widget("button") {
    label = "🎓 Retrain Model with Labels",
    tooltip = "Train ML model using Green Color Labels (Keep) and Red Color Labels (Trash)",
    clicked_callback = function(widget)
        local ok, err = pcall(function()
            log_debug("TRAINING: Button clicked")
            local images = dt.gui.selection()
            if not images or #images == 0 then
                images = {}
                local col_ok, col = pcall(function() return dt.collection end)
                if col_ok and col then
                    for i = 1, #col do
                        table.insert(images, col[i])
                    end
                else
                    for i = 1, #dt.database do
                        table.insert(images, dt.database[i])
                    end
                end
            end
            local total_images = #images
            if total_images == 0 then
                dt.print("Derush Error: No images found to train")
                return
            end

            label_stats_selected.label = string.format("Selected Photos: %d", total_images)

            local job = nil
            pcall(function()
                job = dt.gui.create_job("Derush: Training on " .. total_images .. " photos...")
            end)

            dt.print(string.format("Derush: Scanning Darktable labels across %d photos...", total_images))

            local label_map = {}
            local keep_count = 0
            local trash_count = 0

            for i, img in ipairs(images) do
                if img.green or img.rating > 1 then
                    label_map[img.filename] = "keep"
                    keep_count = keep_count + 1
                elseif img.red or img.rating == -1 then
                    label_map[img.filename] = "trash"
                    trash_count = trash_count + 1
                end
                if i % 100 == 0 or i == total_images then
                    if job then pcall(function() job.percent = (i / total_images) * 0.30 end) end
                end
            end

            if keep_count == 0 or trash_count == 0 then
                if job then pcall(function() job.valid = false end) end
                label_stats_manual.label  = string.format("Manual Labels: %d (%d Keep, %d Trash)", keep_count + trash_count, keep_count, trash_count)
                label_stats_trained.label = "Training: Failed (Insufficient Labels)"
                label_stats_score.label   = "Error: Need >=1 Keep (Green/5⭐) & >=1 Trash (Red/1⭐)"
                dt.print(string.format("Derush Error: Need at least 1 Keep and 1 Trash label! Found: %d Keep, %d Trash.", keep_count, trash_count))
                return
            end

            label_stats_manual.label = string.format("Manual Labels: %d (%d Keep, %d Trash)", keep_count + trash_count, keep_count, trash_count)

            dt.print(string.format("Derush: Extracting features & fitting CatBoost model (%d Keep + %d Trash)...", keep_count, trash_count))

            local json_parts = {}
            for fn, st in pairs(label_map) do
                table.insert(json_parts, string.format('"%s":"%s"', fn, st))
            end
            local labels_json = "{" .. table.concat(json_parts, ",") .. "}"

            local file_paths = {}
            for _, img in ipairs(images) do
                local pth = img.path
                local fn = img.filename
                local full_p = (pth and pth ~= "") and (pth .. "/" .. fn) or fn
                table.insert(file_paths, string.format('"%s"', full_p:gsub("\\", "/")))
            end
            local files_json = "[" .. table.concat(file_paths, ",") .. "]"

            local folder_path = get_collection_root_dir(images)
            if job then pcall(function() job.percent = 0.50 end) end
            log_debug("TRAINING: folder=" .. tostring(folder_path) .. " keep=" .. keep_count .. " trash=" .. trash_count)
            local result = run_derush_command("train", folder_path, labels_json, files_json)
            if job then pcall(function() job.percent = 1.00 end); pcall(function() job.valid = false end) end

            local err_msg = result and result:match('"message":%s*"([^"]+)"')
            local n_samples = tonumber(result and result:match('"n_samples":%s*(%d+)'))
            local n_keep = tonumber(result and result:match('"n_keep":%s*(%d+)'))
            local n_trash = tonumber(result and result:match('"n_trash":%s*(%d+)'))
            local cv_acc = tonumber(result and result:match('"cv_accuracy_mean":%s*([%d%.]+)'))
                or tonumber(result and result:match('"accuracy":%s*([%d%.]+)'))
                or tonumber(result and result:match('"roc_auc":%s*([%d%.]+)'))

            if err_msg and not (cv_acc or n_samples) then
                label_stats_trained.label = "Training: Failed"
                label_stats_score.label   = "Error: " .. err_msg
                dt.print("Derush Error: " .. err_msg)
                return
            end

            if n_samples and n_keep and n_trash then
                label_stats_trained.label = string.format("Training Data: %d unique JPGs (%d Keep, %d Trash)", n_samples, n_keep, n_trash)
            end
            if cv_acc then
                label_stats_score.label = string.format("Model Accuracy: %.1f%%", cv_acc * 100)
            elseif n_samples then
                label_stats_score.label = "Model Accuracy: Ready (Trained)"
            end

            local score_str = cv_acc and string.format(" (Accuracy: %.1f%%)", cv_acc * 100) or ""
            dt.print(string.format("Derush: Trained on %d unique JPGs (%d Keep, %d Trash)%s!",
                n_samples or (keep_count + trash_count),
                n_keep or keep_count,
                n_trash or trash_count,
                score_str))
        end)
        if not ok then
            log_debug("TRAINING ERROR: " .. tostring(err))
        end
    end
}

-- Live Panel Stats Auto-Scanner (scans before button click)
update_panel_stats = function()
    pcall(function()
        local images = dt.gui.selection()
        if not images or #images == 0 then
            images = {}
            local col_ok, col = pcall(function() return dt.collection end)
            if col_ok and col then
                for i = 1, #col do
                    table.insert(images, col[i])
                end
            else
                for i = 1, #dt.database do
                    table.insert(images, dt.database[i])
                end
            end
        end

        local total_images = #images
        if total_images == 0 then
            label_stats_selected.label    = "Photos in View: 0"
            label_stats_manual.label      = "Manual Labels: 0 (0 Keep, 0 Trash)"
            label_stats_predictions.label = "Auto Predicted: 0"
            return
        end

        local keep_count = 0
        local trash_count = 0
        local predicted_count = 0

        for _, img in ipairs(images) do
            -- Check manual Green/5⭐ (Keep) or Red/1⭐/Rejected (Trash)
            if img.green or img.rating > 1 then
                keep_count = keep_count + 1
            elseif img.red or img.rating == -1 then
                trash_count = trash_count + 1
            end

            -- Check if auto-predicted by Derush ML
            local existing_tags = dt.tags.get_tags(img)
            if existing_tags then
                for _, t in ipairs(existing_tags) do
                    if t.name == "derush|predicted" or t.name:find("^derush|score_") then
                        predicted_count = predicted_count + 1
                        break
                    end
                end
            end
        end

        local manual_total = keep_count + trash_count
        label_stats_selected.label    = string.format("Photos in View: %d", total_images)
        label_stats_manual.label      = string.format("Manual Labels: %d (%d Keep, %d Trash)", manual_total, keep_count, trash_count)
        label_stats_predictions.label = string.format("Auto Predicted: %d photos", predicted_count)
    end)
end

-- Auto-trigger stats scanning on Selection Changed or Collection Changed
pcall(function()
    dt.register_event("derush_event_sel", "selection-changed", function()
        update_panel_stats()
    end)
end)

pcall(function()
    dt.register_event("derush_event_col", "collection-changed", function()
        update_panel_stats()
    end)
end)

local map_stars_btn = dt.new_widget("button") {
    label = "⭐ Map Scores to Stars",
    tooltip = "Apply star ratings (1 to 5) based on computed Derush ML scores",
    clicked_callback = function(widget)
        pcall(function()
            local images = dt.gui.selection()
            if not images or #images == 0 then
                images = {}
                local col_ok, col = pcall(function() return dt.collection end)
                if col_ok and col then
                    for i = 1, #col do
                        table.insert(images, col[i])
                    end
                else
                    for i = 1, #dt.database do
                        table.insert(images, dt.database[i])
                    end
                end
            end

            if #images == 0 then
                dt.print("Derush: No images found in current selection/view")
                return
            end

            local mapped_count = 0
            for _, img in ipairs(images) do
                local score = nil
                local existing_tags = dt.tags.get_tags(img)
                if existing_tags then
                    for _, t in ipairs(existing_tags) do
                        local s_str = t.name:match("^derush|score_([%d%.]+)")
                        if s_str then
                            score = tonumber(s_str)
                            break
                        end
                    end
                end
                if not score and img.description then
                    local s_str = img.description:match("Derush Score:%s*([%d%.]+)")
                    if s_str then
                        score = tonumber(s_str)
                    end
                end

                if score then
                    local star_rating = 1
                    if score >= 0.85 then
                        star_rating = 5
                    elseif score >= 0.70 then
                        star_rating = 4
                    elseif score >= 0.50 then
                        star_rating = 3
                    elseif score >= 0.30 then
                        star_rating = 2
                    else
                        star_rating = 1
                    end
                    img.rating = star_rating
                    mapped_count = mapped_count + 1
                end
            end

            dt.print(string.format("Derush: Mapped scores to stars for %d image(s)!", mapped_count))
        end)
    end
}

local sec_actions   = dt.new_widget("section_label") { label = "ML ACTIONS" }
local sec_selection = dt.new_widget("section_label") { label = "COLLECTION SELECTION" }
local sec_model     = dt.new_widget("section_label") { label = "MODEL & TRAINING" }
local sec_scoring   = dt.new_widget("section_label") { label = "SCORING RESULTS" }

local box_selection = dt.new_widget("box") {
    orientation = "vertical",
    label_stats_selected,
    label_stats_manual,
}

local box_model = dt.new_widget("box") {
    orientation = "vertical",
    label_stats_score,
    label_stats_trained,
}

local box_scoring = dt.new_widget("box") {
    orientation = "vertical",
    label_stats_predictions,
    label_stats_scores_detail,
}

local widget_box = dt.new_widget("box") {
    orientation = "vertical",
    sec_actions,
    predict_btn,
    train_btn,
    map_stars_btn,
    sec_selection,
    box_selection,
    sec_model,
    box_model,
    sec_scoring,
    box_scoring,
}

dt.register_lib("derush_panel", "Photo-Derush", true, false, {
    [dt.gui.views.lighttable] = {"DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100}
}, widget_box)

-- Initial scan on plugin load
update_panel_stats()
