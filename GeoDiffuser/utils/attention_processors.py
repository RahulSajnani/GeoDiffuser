import torch
import abc
from typing import Optional, Union, Tuple, List, Callable, Dict
from GeoDiffuser.utils.attention_sharing import *
from diffusers.models.attention_processor import USE_PEFT_BACKEND


def register_attention_control_diffusers(model, controller, transform_coords = None):
    
    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        # print(name)

        cross_att_count += 1
        attn_processor = EditProcessor(transform_coords, controller, place_in_unet)
        # if name.startswith("up_blocks.3") or name.startswith("up_blocks.2"):
        # if name.startswith("up_blocks.3") or name.startswith("up_blocks.2") or name.startswith("down_blocks.0") or name.startswith("down_blocks.1"):
        # if 
        if 1:
            attn_procs[name] = attn_processor
        else:
            attn_procs[name] = VanillaAttentionProcessor()

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


def set_attn_processor_for_edit(model, perform_edit=True, coords_base=(2,3), coords_edit = (3,4), use_cfg = True):
    # Setting attn processor to enable / disable editing
    for name in model.unet.attn_processors.keys():

        # if name.startswith("up_blocks.3") or name.startswith("up_blocks.2"):
        if 1:
        # if name.startswith("up_blocks.3") or name.startswith("up_blocks.2") or name.startswith("down_blocks.0") or name.startswith("down_blocks.1"):

            model.unet.attn_processors[name].perform_edit = perform_edit
            model.unet.attn_processors[name].controller.coords_base = coords_base
            model.unet.attn_processors[name].controller.coords_edit = coords_edit
            model.unet.attn_processors[name].controller.use_cfg = use_cfg

class VanillaAttentionProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self,):
        pass

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(None, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class EditProcessor:
    def __init__(self, transform_coords, controller, place_in_unet = "down", perform_edit = True, coords_base=(2, 3), coords_edit=(3,4), use_cfg=True):
        # self.attention_probs = None
        self.transform_coords = transform_coords
        self.place_in_unet = place_in_unet
        self.perform_edit = perform_edit
        self.controller = controller

        self.controller.use_cfg = use_cfg
        self.controller.coords_base = coords_base
        self.controller.coords_edit = coords_edit

    
    # @torch.enable_grad()
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask = None,
        temb = None,
        scale: float = 1.0,
    ) -> torch.Tensor:

        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        # print(attn.scale, attn.rescale_output_factor)
        # args = (scale,)
        # args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)
        is_cross = True
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            is_cross = False
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)


        
        if self.perform_edit:
            hidden_states = self.controller(query, key, value, is_cross = is_cross, place_in_unet = self.place_in_unet, transform_coords = self.transform_coords, scale = attn.scale,  mask = None)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states




class AttentionGeometryEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t, transform_coords=None):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, transform_coords)
        return x_t
    
    def replace_cross_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = False, scale = None, mask = None, coords_base = None, coords_edit = None, old_attention_map = None, old_attention_out = None):
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        t_coords_m = reshape_transform_coords(transform_coords, in_mat_shape=self.image_mask.shape).tile(self.image_mask.shape[0], 1, 1, 1).type_as(q_base)
        
        if self.mask_new_warped is None:
            mask_new_warped = binarize_tensor(warp_grid_edit(self.image_mask[:, None], t_coords_m, padding_mode='zeros', align_corners=True, mode=MODE))
            self.mask_new_warped = mask_new_warped.detach()
        else:
            mask_new_warped = self.mask_new_warped.detach()


        amodal_mask = self.amodal_mask.detach()




        # print(amodal_mask.shape, mask_new_warped.shape, " amodal mask shape vs mask new warped shape")

        mask_warp = binarize_tensor(self.image_mask)[:, None]
        # print(amodal_mask.shape, mask_warp.shape)
        amodal_mask = amodal_mask.type_as(mask_warp)


        b, f = mask_new_warped.shape[:2]

        mask_new_warped = reshape_attention_mask(mask_new_warped, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]# b-1, 1, h, w
        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        amodal_mask = reshape_attention_mask(amodal_mask, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))
        amodal_mask = binarize_tensor(amodal_mask.type_as(mask_new_warped) - mask_new_warped).detach()
        
        mask_intersection = binarize_tensor((mask_new_warped + amodal_mask) * mask_warp, 0.5)
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp - mask_intersection), 0.5)


        h, w = mask_warp.shape[-2:]

        distance_grid = DISTANCE_CLASS.get_coord_distance(h).detach()
        # print(distance_grid.shape)
        # exit()
        
        # print(mask_warp.shape, h *w, 32**2)
        # if (h * w) >= 64**2:
        #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)
        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_new_warped) - binarize_tensor(mask_1_empty + mask_new_warped + amodal_mask))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
            self.mask_1_empty = mask_1_empty.detach()
        
        
        b, f, _, D = q_base.shape
        
        
        
        # Transform q_base to edit image        
        q_edit_base = q_base.permute(0, 1, -1, 2).reshape(b, f, D, h, w).reshape(-1, D, h, w)
        t_coords_q = reshape_transform_coords(transform_coords, in_mat_shape=q_edit_base.shape).tile(q_edit_base.shape[0], 1, 1, 1).type_as(q_edit_base)
        
        # Transform locations
        q_edit_base = q_edit_base * (1.0 - mask_new_warped) + mask_new_warped * warp_grid_edit(q_edit_base, t_coords_q, padding_mode='zeros', align_corners=True, mode=MODE)
        
        q_edit_base = q_edit_base.reshape(b, f, D, h, w).reshape(b, f, D, -1).permute(0, 1, -1, 2).reshape(-1, h*w, D)

        
        

        edit_base_att = compute_attention(q_edit_base, k_base.reshape(b*f, -1, D), scale, mask)
        edit_out = torch.bmm(edit_base_att, v_base.detach().reshape(b*f, -1, D))[None].detach()
        
        
        # edit_out = perform_attention(q_edit_base.reshape(b, f, -1, D), k_base.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale).reshape(b*f, -1 ,D).detach()[None]
        
        
        
        b, f, _, D = q_edit.shape

        # if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):
        #     replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask, fg_mask_warp = mask_new_warped, fg_mask = mask_warp)
        #     replace_out = torch.bmm(replace_out_att, v_base.detach().reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D

        # else:
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        replace_out = torch.bmm(replace_out_att, v_base.detach().reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        # replace_out = perform_attention(q_edit.reshape(b, f, -1, D), k_edit.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale)
        


        # if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):


        #     replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        #     replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_inpaint = mask_1_empty[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]
        mask_edit = mask_new_warped[0, 0].reshape(-1)[None, None, :, None]
        amodal_mask = amodal_mask[0, 0].reshape(-1)[None, None, :, None]


        # print(mask_inpaint.shape, replace_out.shape)
        b, f, _, D = replace_out.shape



        if self.use_cfg and self.store_attention_maps:
            # print(q_edit.shape)
            self.attn_store(replace_out_att, is_cross=True, place_in_unet = place_in_unet)





        if mask_inpaint.shape[2] >= 32 ** 2 and (not self.use_cfg):
            interpolated_features, interpolation_weights = interpolate_from_mask(edit_out, mask_edit, distance_grid)
            


            # print(interpolated_features.shape, edit_out.shape)
            interpolated_features[:, :, mask_edit[0, 0, :, 0] > 0.5] = edit_out[:, :, mask_edit[0, 0, :, 0] > 0.5].detach().type_as(interpolated_features)
            # print(interpolated_features.shape, edit_out.shape)
            interpolated_features = smooth_attention_features(interpolated_features)
            # exit()
            


            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint * torch.ones_like(replace_out.detach())) + 1e-8))
            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint.expand_as(replace_out)) + 1e-8))

            # dissimilar_loss = get_correlation_loss_stitch(compute_attention(q_edit.reshape(b*f, -1, D), q_base.detach().reshape(b*f, -1, D), scale, mask), mask_inpaint, mask_wo_edit, mask_inpaint)



            # print(sim_loss)

            # with torch.no_grad():
            #     # Do not get min distance from regions that are in the background
            #     distance_weights = distance_grid.type_as(mask_wo_edit) + 100 * mask_wo_edit[:1, :1, :, 0]
            #     distance_weights = 1.0 - torch.exp(-2 * torch.min(distance_weights, -1).values) # 1, hw
            # sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit * distance_weights[:, None, :, None]) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach()) * distance_weights[:, None, :, None]) + 1e-8))

            # print(sim_loss)
            # exit()
            sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach())) + 1e-8))
            # movement_loss = (torch.sum((1 -  torch.exp((-torch.abs(edit_out.detach() - replace_out)))) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out.detach())) + 1e-8))


            movement_loss = torch.sum(torch.abs(edit_out.detach() - replace_out) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out.detach())) + 1e-8)
            
            
            with torch.no_grad():
                if self.use_cfg:
                    ah = q.shape[0] // (2 * self.batch_size)
                else:
                    ah = q.shape[0] // (self.batch_size)

                base_att = old_attention_map[coords_base[0] * ah: coords_base[1] * ah]
                # base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)

            correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], base_att.permute(0, -1, 1).detach())

            correlation_inpaint = correlation * mask_inpaint[..., 0]
            correlation_wo_edit = correlation * mask_wo_edit[..., 0]

            m_c_inpaint = torch.max(correlation_inpaint, -1)
            m_c_wo_edit = torch.max(correlation_wo_edit, -1)

            p_correlation_inpaint, d_inpaint = m_c_inpaint.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_inpaint.indices]
            p_correlation_wo_edit, d_wo_edit = m_c_wo_edit.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_wo_edit.indices]


            with torch.no_grad():
                d_weight = torch.exp(-d_wo_edit.detach())

            dissimilar_loss = torch.sum(d_weight.detach() * (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))) / (torch.sum(mask_inpaint) * f + 1e-8)
            del m_c_inpaint, m_c_wo_edit


            # correlation_amodal = torch.bmm(replace_out_att[:, amodal_mask[0, 0, :, 0] > 0.5], base_att[:, mask_inpaint[0, 0, :, 0] > 0.5].permute(0, -1, 1).detach())

            # correlation_amodal_loss = torch.sum(-torch.log(torch.max(correlation_amodal, -1).values + 1e-8)) / (torch.sum(torch.ones_like(correlation_amodal)) + 1e-8)



            amodal_loss = (torch.sum((torch.abs(interpolated_features.detach() - replace_out)) * interpolation_weights[..., None] * amodal_mask) / (torch.sum(interpolation_weights[..., None] * amodal_mask * torch.ones_like(replace_out.detach())) + 1e-8))     

            # amodal_loss = 0.0 * movement_loss
            # correlation_amodal = torch.bmm(replace_out_att[:, amodal_mask[0, 0, :, 0] > 0.5], edit_base_att[:, :, ].permute(0, -1, 1).detach())


            # amodal_loss = 


            if mask_inpaint.shape[2] <= 32 ** 2:
                amodal_loss = 0.0 * movement_loss



            dissociate_loss, att_loss, dissociate_loss_2 = 0, 0, 0

            smoothness_loss, _, _ = get_smoothness_loss(replace_out)

            # if mask_inpaint.shape[2] >= 64 ** 2:
            #     smoothness_loss, _, _ = get_smoothness_loss(replace_out)
            # else: 
            #     smoothness_loss = 0.0 * movement_loss

            lw = self.loss_weight_dict["cross"]
            self.loss += lw["sim"] * sim_loss + lw["movement"] * movement_loss + lw["removal"] * dissimilar_loss + lw["smoothness"] * smoothness_loss + lw["amodal"] * amodal_loss

            loss_log_dict = {"sim": sim_loss, "movement": movement_loss, "removal": dissimilar_loss, "smoothness": smoothness_loss, "amodal": amodal_loss}
            self.loss_log_dict["cross"] = update_loss_log_dict(self.loss_log_dict["cross"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1
            
            

            # self.loss += (0.0 * dissociate_loss + 0.0 * att_loss + 10.0 * sim_loss + 10.0 * movement_loss + 0.0 *  dissociate_loss_2 + 8.0 * dissimilar_loss) / 3



        
        # edit_out[:, :, amodal_mask[0, 0, :, 0] > 0.5] = interpolated_features[:, :, amodal_mask[0, 0, :, 0] > 0.5].detach().type_as(edit_out)
        # mask_edit = binarize_tensor(mask_edit + amodal_mask)
        
        if (self.cur_step < int(self.num_steps * self.obj_edit_step)):
            out_return = edit_out.detach() * (mask_edit) + replace_out * (1.0 - mask_edit) #mask_inpaint + replace_out * mask_wo_edit
        # elif mask_inpaint.shape[2] <= 16 ** 2:
        #     out_return = edit_out.detach() * (mask_edit) + replace_out * (1.0 - mask_edit)
        else:
            out_return = replace_out
            # print("identity")
            # out_return = replace_out_identity * (1.0 - mask_edit) + edit_out.detach() * mask_edit
            # out_return = replace_out_identity * (mask_edit) + replace_out_identity * mask_inpaint + replace_out * mask_wo_edit
        
        return out_return

        
        
        
    def replace_self_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = None, scale = None, mask = None, coords_base = None, coords_edit = None, old_attention_map = None, old_attention_out = None):
        
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        t_coords_m = reshape_transform_coords(transform_coords, in_mat_shape=self.image_mask.shape).tile(self.image_mask.shape[0], 1, 1, 1).type_as(q_base)
        

        if self.mask_new_warped is None:
            mask_new_warped = binarize_tensor(warp_grid_edit(self.image_mask[:, None], t_coords_m, padding_mode='zeros', align_corners=True, mode=MODE))
            self.mask_new_warped = mask_new_warped.detach()

        else:
            mask_new_warped = self.mask_new_warped.detach()


        amodal_mask = self.amodal_mask.detach()

        # amodal_mask = binarize_tensor(self.amodal_mask.type_as(mask_new_warped) - mask_new_warped).detach()

        # print(amodal_mask.shape, mask_new_warped.shape, " amodal mask shape vs mask new warped shape")
            

        mask_warp = binarize_tensor(self.image_mask.type_as(q_base))[:, None]
        # print(amodal_mask.shape, mask_warp.shape)
        # exit()

        amodal_mask = amodal_mask.type_as(mask_warp)
        b, f = mask_new_warped.shape[:2]

        mask_new_warped = reshape_attention_mask(mask_new_warped, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]# b-1, 1, h, w
        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        amodal_mask = reshape_attention_mask(amodal_mask, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))
        amodal_mask = binarize_tensor(amodal_mask.type_as(mask_new_warped) - mask_new_warped).detach()
        
        
        
        mask_intersection = binarize_tensor((mask_new_warped + amodal_mask) * mask_warp, 0.5)
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp - mask_intersection), 0.5)
        
        h, w = mask_warp.shape[-2:]
        distance_grid = DISTANCE_CLASS.get_coord_distance(h).detach()
        
        # if (h * w) >= 64**2:
        #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)
        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_new_warped) - (mask_1_empty + mask_new_warped))
        
        
        
        b, f, _, D = q_base.shape
        # Transform q_base to edit image        
        q_edit_base = q_base.permute(0, 1, -1, 2).reshape(b, f, D, h, w).reshape(-1, D, h, w)
        t_coords_q = reshape_transform_coords(transform_coords, in_mat_shape=q_edit_base.shape).tile(q_edit_base.shape[0], 1, 1, 1).type_as(q_edit_base)
        
        # Transform locations
        q_edit_base = q_edit_base * (1.0 - mask_new_warped) + mask_new_warped * warp_grid_edit(q_edit_base, t_coords_q, padding_mode='zeros', align_corners=True, mode=MODE)
        
        q_edit_base = q_edit_base.reshape(b, f, D, h, w).reshape(b, f, D, -1).permute(0, 1, -1, 2).reshape(-1, h*w, D)
        
        
        edit_base_att = compute_attention(q_edit_base, k_base.reshape(b*f, -1, D), scale, mask)
        edit_out = torch.bmm(edit_base_att, v_base.detach().reshape(b*f, -1, D))[None].detach()
        
        
        b, f, _, D = q_edit.shape

        # q_edit_adain = adain(q_edit.reshape(b*f, -1, D), q_base.detach().reshape(b*f, -1, D))

        # if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask, fg_mask_warp = mask_new_warped, fg_mask = mask_warp, inpaint_mask = mask_wo_edit)
        replace_out = torch.bmm(replace_out_att, v_base.detach().reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        
        # else:

        #     replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask)
        #     replace_out = torch.bmm(replace_out_att, v_base.detach().reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        

        if self.use_cfg and self.store_attention_maps: 
            # print(q_edit.shape)

            self.attn_store(replace_out_att, is_cross=False, place_in_unet = place_in_unet)

        # edit_out = perform_attention(q_edit_base.reshape(b, f, -1, D), k_base.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale).reshape(b*f, -1 ,D).detach()[None]
        
        
        
        # b, f, _, D = q_edit.shape
        # replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        # replace_out = torch.bmm(replace_out_att, v_base.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        # replace_out = perform_attention(q_edit.reshape(b, f, -1, D), k_edit.reshape(b, f, -1, D), v_base.reshape(b, f, -1, D), scale = scale)


        # if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):

        #     replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
        #     replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_inpaint = mask_1_empty[0, 0].reshape(-1)[None, None, :, None]
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]
        mask_edit = mask_new_warped[0, 0].reshape(-1)[None, None, :, None]
        amodal_mask = amodal_mask[0, 0].reshape(-1)[None, None, :, None]

        
        if mask_inpaint.shape[2] >= 32 ** 2:
            self.mask_inpaint = mask_1_empty[0, 0].detach().clone()
        
        

        # if (self.cur_step < int(self.num_steps * 0.6)):
        #     edit_out = smooth_attention_features(edit_out.detach())



        if mask_inpaint.shape[2] >= 32 ** 2 and (not self.use_cfg):
            interpolated_features, interpolation_weights = interpolate_from_mask(edit_out, mask_edit, distance_grid)

            interpolated_features[:, :, mask_edit[0, 0, :, 0] > 0.5] = edit_out[:, :, mask_edit[0, 0, :, 0] > 0.5].detach().type_as(interpolated_features)
            # print(interpolated_features.shape, edit_out.shape)
            interpolated_features = smooth_attention_features(interpolated_features)

            # sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach())) + 1e-8))

            
            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint * torch.ones_like(replace_out.detach())) + 1e-8))

            with torch.no_grad():

                if self.use_cfg:
                    ah = q.shape[0] // (2 * self.batch_size)
                else:
                    ah = q.shape[0] // (self.batch_size)

                base_att = old_attention_map[coords_base[0] * ah: coords_base[1] * ah]
                # base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)
                # print(base_att.shape)


            # print(mask_inpaint.shape, replace_out_att.shape)
            # print(torch.sum((mask_inpaint[0, 0, :, 0] > 0.5) * 1.0))

            # exit()
            correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], base_att.permute(0, -1, 1).detach())

            correlation_inpaint = correlation * mask_inpaint[..., 0]
            correlation_wo_edit = correlation * mask_wo_edit[..., 0]

            # p_correlation_inpaint = torch.max(correlation_inpaint, -1).values
            # p_correlation_wo_edit = torch.max(correlation_wo_edit, -1).values

            # dissimilar_loss = torch.sum(-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4)) / (torch.sum(mask_inpaint) * f + 1e-8)

            m_c_inpaint = torch.max(correlation_inpaint, -1)
            m_c_wo_edit = torch.max(correlation_wo_edit, -1)

            p_correlation_inpaint, d_inpaint = m_c_inpaint.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_inpaint.indices]
            p_correlation_wo_edit, d_wo_edit = m_c_wo_edit.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_wo_edit.indices]

            # print(d_inpaint.max(), d_inpaint.min(), d_wo_edit.max(), d_wo_edit.min())
            
            # print("before: ", torch.log(p_correlation_inpaint + 1e-4).min(), torch.log(p_correlation_inpaint + 1e-4).max())
            # print("after: ", torch.exp(-15 * d_wo_edit.detach()).max(), (torch.exp(-15 * d_wo_edit.detach()) * torch.log(p_correlation_inpaint + 1e-4)).min(), (torch.exp(-15 * d_wo_edit.detach()) * torch.log(p_correlation_inpaint + 1e-4)).max())

            # d_before = (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))
            # d_after = (-torch.log(p_correlation_wo_edit + 1e-4) + torch.exp(-15 * d_wo_edit.detach()) * torch.log(p_correlation_inpaint + 1e-4))
            
            
            # print("+++++++++++++++++++++++++++++++++++++++++++++++")
            # print("before: ", d_before.min(), d_before.max())
            # print("after: ", d_after.min(), d_after.max())

            

            with torch.no_grad():
                d_weight = torch.exp(-d_wo_edit.detach())
                
            dissimilar_loss = torch.sum(d_weight.detach() * (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))) / (torch.sum(mask_inpaint) * f + 1e-8)

            del m_c_inpaint, m_c_wo_edit


            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint.expand_as(replace_out)) + 1e-8))
            # dissimilar_loss = get_correlation_loss_stitch(replace_out_att, mask_inpaint, mask_wo_edit, mask_inpaint)

            # with torch.no_grad():
            #     # Do not get min distance from regions that are in the background
            #     distance_weights = distance_grid.type_as(mask_wo_edit) + 100 * mask_wo_edit[:1, :1, :, 0]
            #     distance_weights = 1.0 - torch.exp(-2 * torch.min(distance_weights, -1).values) # 1, hw
                
            # sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit * distance_weights[:, None, :, None]) / (torch.sum(mask_wo_edit * torch.ones_like(replace_out.detach()) * distance_weights[:, None, :, None]) + 1e-8))

            sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit.expand_as(replace_out)) + 1e-8))

            # movement_loss = (torch.sum((1 -  torch.exp((-torch.abs(edit_out.detach() - replace_out)))) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out.detach())) + 1e-8))
            movement_loss = torch.sum(torch.abs(edit_out.detach() - replace_out) * mask_edit) / (torch.sum(mask_edit * torch.ones_like(replace_out.detach())) + 1e-8)


            

            # correlation_amodal = torch.bmm(replace_out_att[:, amodal_mask[0, 0, :, 0] > 0.5], base_att[:, mask_inpaint[0, 0, :, 0] > 0.5].permute(0, -1, 1).detach())

            # correlation_amodal_loss = torch.sum(-torch.log(torch.max(correlation_amodal, -1).values + 1e-8)) / (torch.sum(torch.ones_like(correlation_amodal)) + 1e-8)


            # interpolated_features, interpolation_weights = interpolate_from_mask(edit_out, mask_edit, distance_grid)
            amodal_loss = (torch.sum((torch.abs(interpolated_features.detach() - replace_out)) * interpolation_weights[..., None] * amodal_mask) / (torch.sum(interpolation_weights[..., None] * amodal_mask * torch.ones_like(replace_out.detach())) + 1e-8))        


            # print(interpolated_features.shape, interpolation_weights.shape, replace_out.shape)
            # print(amodal_loss)


            if mask_inpaint.shape[2] <= 32 ** 2:
                amodal_loss = 0.0 * movement_loss

            
            
            dissociate_loss, att_loss, dissociate_loss_2 = 0, 0, 0
            smoothness_loss, _, _ = get_smoothness_loss(replace_out)

            # if mask_inpaint.shape[2] >= 64 ** 2:
            #     smoothness_loss, _, _ = get_smoothness_loss(replace_out)
            # else:
            #     smoothness_loss = 0.0 * movement_loss

            lw = self.loss_weight_dict["self"]
            self.loss += lw["sim"] * sim_loss + lw["movement"] * movement_loss + lw["removal"] * dissimilar_loss + lw["smoothness"] * smoothness_loss + lw["amodal"] * amodal_loss

            
            loss_log_dict = {"sim": sim_loss, "movement": movement_loss, "removal": dissimilar_loss, "smoothness": smoothness_loss, "amodal": amodal_loss}
            self.loss_log_dict["self"] = update_loss_log_dict(self.loss_log_dict["self"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1
            
            

            # self.loss += (0.0 * dissociate_loss + 0.0 * att_loss + 55.0 * sim_loss + 20.0 * movement_loss + 0.0 * dissociate_loss_2 + 13.0 * dissimilar_loss) / 3

        # mask_edit = binarize_tensor(mask_edit + amodal_mask)

        # edit_out[:, :, amodal_mask[0, 0, :, 0] > 0.5] = interpolated_features[:, :, amodal_mask[0, 0, :, 0] > 0.5].detach().type_as(edit_out)
        # mask_edit = binarize_tensor(mask_edit + amodal_mask)

        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = edit_out.detach() * (mask_edit) + replace_out * (1.0 - mask_edit) #mask_inpaint + replace_out * mask_wo_edit
        # elif mask_inpaint.shape[2] <= 16 ** 2:
        #     out_return = edit_out.detach() * (mask_edit) + replace_out * (1.0 - mask_edit)
        else:
            out_return = replace_out

            # print("identity")
            # out_return = replace_out_identity * (1.0 - mask_edit) + edit_out.detach() * mask_edit
            # out_return = replace_out_identity * (mask_edit) + replace_out_identity * mask_inpaint + replace_out * mask_wo_edit

        return out_return

    def initialize_loss_log_dict(self):

        self.loss_log_dict = {"self": {"sim": 0.0, "movement": 0.0, "removal": 0.0, "smoothness": 0.0},
                             "cross": {"sim": 0.0, "movement": 0.0, "removal": 0.0, "smoothness": 0.0}, 
                             "num_layers": 0}

    
    def forward(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale = None, mask = None):


        if self.use_cfg:
            h = q.shape[0] // (2 * self.batch_size)
        else:
            h = q.shape[0] // self.batch_size


        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            attn = compute_attention(q[:self.coords_base[-1] * h], k[:self.coords_base[-1] * h], scale, mask)
            out = torch.bmm(attn, v[:self.coords_base[-1] * h])
        else:
            attn = compute_attention(q, k, scale, mask)
            out = torch.bmm(attn, v)

        


        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                out_edit = self.replace_cross_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty = False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit, old_attention_map = attn, old_attention_out = out)
                
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

            else:
                out_edit = self.replace_self_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty=False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit, old_attention_map = attn, old_attention_out = out)
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])
        

        return out
    
    
    def initialize_default_loss_weights(self):
        self.loss_weight_dict = self.default_loss_weights

        # old
        # self.loss_weight_dict = {"self": {"sim": 40, "movement": 1.5, "removal": 1.67, "smoothness": 45.0},
        #                         "cross": {"sim": 25, "movement": 1.34, "removal": 1.6, "smoothness": 20.0}}
        # Increase movement loss
        # reduce background preservation loss
        # From remover : )
        # self.loss_weight_dict = {"self": {"sim": 0.97, "scale": 75, "smoothness": 85.0},
        # "cross": {"sim": 0.92, "scale": 45.0, "smoothness": 45.0}}


    # def store_default_loss_weights(self):
    #     self.default

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer = None,
                local_blend = None, controller = None, image_mask = None, empty_scale = 0.2, use_all = True, obj_edit_step = 0.0):
        super(AttentionGeometryEdit, self).__init__()
        if equalizer is not None:
            eq = get_equalizer(prompts[1], equalizer["words"], equalizer["values"])
            self.equalizer = eq.to(DEVICE)
        self.prev_controller = controller
        self.last_cross_mask = None
        
        if image_mask is not None:
            image_mask = torch.from_numpy(image_mask[None])
            # print(image_mask.shape)
            image_mask = image_mask.tile((len(prompts), 1, 1))
            self.image_mask = image_mask
        self.thre = 0.00001
        self.empty_scale = empty_scale
        self.use_all = use_all
        self.loss = 0.0

        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(DEVICE)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.mask_inpaint = None
        self.obj_edit_step = obj_edit_step
        self.num_steps = num_steps

        self.mask_new_warped = None
        self.mask_wo_edit = None
        self.mask_1_empty = None
        self.amodal_mask = None

        self.coords_base = (2, 3)
        self.coords_edit = (3, 4)
        self.use_cfg = True


        # self.loss_weight_dict = {"self": {"sim": 0.74, "scale": 25.0, "removal": 4.34, "smoothness": 0.0},
        #                         "cross": {"sim": 0.5, "scale": 6.67, "removal": 2.67, "smoothness": 0.0}}
        

        self.default_loss_weights = {"self": {"sim": 110, "movement": 13.5, "removal": 1.67, "smoothness": 35.0, "amodal": 80.5},
                                "cross": {"sim": 60, "movement": 6.34, "removal": 1.6, "smoothness": 20.0, "amodal": 3.5}}
        self.loss_log_dict = None
        self.loss_weight_dict = None
        self.initialize_loss_log_dict()
        self.store_attention_maps = False
        self.initialize_default_loss_weights()



class AttentionGeometryRemover(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t, transform_coords=None):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, transform_coords)
        return x_t
    
    def replace_cross_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = False, scale = None, mask = None, coords_base = None, coords_edit = None, old_attention_map = None, old_attention_out = None):
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)

        if self.use_cfg:
            h_head = q.shape[0] // (2 * self.batch_size)
        else:
            h_head = q.shape[0] // self.batch_size
        
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        
        mask_warp = binarize_tensor(self.image_mask)[:, None]
        b, f = mask_warp.shape[:2]

        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp), 0.5)

        h, w = mask_warp.shape[-2:]
        distance_grid = DISTANCE_CLASS.get_coord_distance(h).detach()

        # if (h * w) >= 64**2:
        # #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)

        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_warp) - (mask_1_empty))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
            self.mask_1_empty = mask_1_empty.detach()
        
        
        b, f, _, D = q_base.shape

        edit_base_att = old_attention_map[coords_base[0] * h_head:coords_base[1]*h_head].detach()
        edit_out = old_attention_out[coords_base[0] * h_head:coords_base[1]*h_head][None].detach()
        # edit_base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)
        # edit_out = torch.bmm(edit_base_att, v_base.reshape(b*f, -1, D))[None].detach()
        
        
        b, f, _, D = q_edit.shape
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask)
        replace_out = torch.bmm(replace_out_att, v_base.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        



        # print(edit_base_att.shape, replace_out_att.shape)

        
        # if self.cur_step < int(self.num_steps * self.obj_edit_step):
        if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):

            replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
            replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_inpaint = mask_1_empty[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]


        b, f, _, D = replace_out.shape



        if mask_inpaint.shape[2] >= 32 ** 2 and (not self.use_cfg):



            # dissimilar_loss = - (torch.sum(torch.min((torch.sum(torch.abs(replace_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5][:, :, :, None] - edit_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5][:, :, None].detach()), -1)), -1).values) / (torch.sum(mask_inpaint.expand_as(replace_out)) * 8 + 1e-8)) # b, 8, M_#

            # print(dissimilar_loss, " cross")
            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint.expand_as(replace_out)) + 1e-8))
            sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit.expand_as(replace_out)) + 1e-8))

            # correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], edit_base_att.permute(0, -1, 1).detach())
            
            # # loss_correlation = torch.sum(correlation  * mask_inpaint[..., 0] - correlation  * mask_wo_edit[..., 0]) / (torch.sum(mask_inpaint) * h * w * f + 1e-8)
            # loss_correlation = torch.sum(torch.max(correlation * mask_inpaint[..., 0], -1).values - torch.max(correlation  * mask_wo_edit[..., 0], -1).values) / (torch.sum(mask_inpaint) * f + 1e-8)

            correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], edit_base_att.permute(0, -1, 1).detach())

            correlation_inpaint = correlation * mask_inpaint[..., 0]
            correlation_wo_edit = correlation * mask_wo_edit[..., 0]

            # p_correlation_inpaint = torch.max(correlation_inpaint, -1).values
            # p_correlation_wo_edit = torch.max(correlation_wo_edit, -1).values

            # loss_correlation = torch.sum(-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4)) / (torch.sum(mask_inpaint) * f + 1e-8)

            m_c_inpaint = torch.max(correlation_inpaint, -1)
            m_c_wo_edit = torch.max(correlation_wo_edit, -1)

            p_correlation_inpaint, d_inpaint = m_c_inpaint.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_inpaint.indices]
            p_correlation_wo_edit, d_wo_edit = m_c_wo_edit.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_wo_edit.indices]


            with torch.no_grad():
                d_weight = torch.exp(-d_wo_edit.detach())

            loss_correlation = torch.sum(d_weight.detach() * (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))) / (torch.sum(mask_inpaint) * f + 1e-8)
            del m_c_inpaint, m_c_wo_edit

            smoothness_loss, _, _ = get_smoothness_loss(replace_out)
            # print(smoothness_loss)
            
            

    # b, h*w, h*w
            # print(loss_correlation, "Cross")        
            dissociate_loss, att_loss, dissociate_loss_2, movement_loss = 0, 0, 0, 0.0

            lw = self.loss_weight_dict["cross"]
            self.loss += (lw["sim"] * sim_loss + lw["removal"]* loss_correlation) + lw["smoothness"] * smoothness_loss


            loss_log_dict = {"sim": sim_loss, "removal": loss_correlation, "smoothness": smoothness_loss}
            self.loss_log_dict["cross"] = update_loss_log_dict(self.loss_log_dict["cross"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1

            # self.loss += 0.5 * (0.0 * dissociate_loss + 0.0 * att_loss + 200.0 * sim_loss + 0.0 * movement_loss + 0.0 *  dissociate_loss_2 + 0.0 * dissimilar_loss + 20.0 * loss_correlation + 200 * smoothness_loss) / 3


        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = replace_out * mask_inpaint + replace_out * mask_wo_edit
        else:
            out_return = replace_out_identity * mask_inpaint + replace_out * mask_wo_edit
        

        return out_return

        
        
        
    def replace_self_attention(self, q, k, v, place_in_unet, transform_coords=None, add_empty = None, scale = None, mask = None, coords_base = None, coords_edit = None, old_attention_map = None, old_attention_out = None):
        
        
        q_base, k_base, v_base, q_edit, k_edit, v_edit = get_base_edit_qkv(q, k, v, self.batch_size, coords_base = coords_base, coords_edit = coords_edit, use_cfg = self.use_cfg)
        
        if self.use_cfg:
            h_head = q.shape[0] // (2 * self.batch_size)
        else:
            h_head = q.shape[0] // self.batch_size
        
        self.image_mask = self.image_mask.type_as(q_base).detach()
        mask_warp = binarize_tensor(self.image_mask)[:, None]
        b, f = mask_warp.shape[:2]
        
        mask_warp = reshape_attention_mask(mask_warp, in_mat_shape=(1, int(np.sqrt(q_base.shape[2]))))[1:]
        # mask to inpaint
        mask_1_empty = binarize_tensor((mask_warp), 0.5)
        h, w = mask_warp.shape[-2:]
        distance_grid = DISTANCE_CLASS.get_coord_distance(h).detach()

        # if (h * w) >= 64**2:
        # #     # print("smoothing")
        #     mask_1_empty = binarize_tensor(smooth_mask(mask_1_empty), 0.5)

        
        mask_wo_edit = binarize_tensor(torch.ones_like(mask_warp) - (mask_1_empty))


        if (h * w) >= 32**2:
            self.mask_wo_edit = mask_wo_edit.detach()
            self.mask_1_empty = mask_1_empty.detach()
        
        
        b, f, _, D = q_base.shape


        edit_base_att = old_attention_map[coords_base[0] * h_head:coords_base[1]*h_head].detach()
        edit_out = old_attention_out[coords_base[0] * h_head:coords_base[1]*h_head][None].detach()
        # edit_base_att = compute_attention(q_base.reshape(b*f, -1, D), k_base.reshape(b*f, -1, D), scale, mask)
        # edit_out = torch.bmm(edit_base_att, v_base.reshape(b*f, -1, D))[None].detach()
        
        
        b, f, _, D = q_edit.shape
        replace_out_att = compute_attention(q_edit.reshape(b*f, -1, D), k_base.detach().reshape(b*f, -1, D), scale, mask)
        replace_out = torch.bmm(replace_out_att, v_base.reshape(b*f, -1, D)).reshape(b, f, -1, D) # 1, f, h*w, D
        

        # if self.cur_step < int(self.num_steps * self.obj_edit_step):


        if not (self.cur_step < int(self.num_steps * self.obj_edit_step)):
            replace_out_identity_att = compute_attention(q_edit.reshape(b*f, -1, D), k_edit.reshape(b*f, -1, D), scale, mask)
            replace_out_identity = torch.bmm(replace_out_identity_att, v_edit.reshape(b*f, -1, D)).reshape(b, f, -1, D) 
        
        
        edit_out = edit_out.tile(replace_out.shape[0], 1, 1, 1) # b, f, h*w, D
        
        
        mask_inpaint = mask_1_empty[0, 0].reshape(-1)[None, None, :, None] # 1, 1, h*w, 1
        mask_wo_edit = mask_wo_edit[0, 0].reshape(-1)[None, None, :, None]


        b, f, _, D = replace_out.shape


        # print(edit_out.shape, replace_out.shape, mask_inpaint.shape)
        # print(replace_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5].shape)

        # dissimilar_loss = - (torch.sum(torch.min((torch.sum(torch.abs(replace_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5][:, :, :, None] - edit_out[:, :, mask_inpaint[0, 0, :, 0] > 0.5][:, :, None].detach()), -1)), -1).values) / (torch.sum(mask_inpaint.expand_as(replace_out)) * 8 + 1e-8)) # b, 8, M_#


        if mask_inpaint.shape[2] >= 32 ** 2 and (not self.use_cfg):
            # dissimilar_loss = -(torch.sum(torch.sum((((torch.abs(edit_out.detach() - replace_out)))), -1)[..., None] * mask_inpaint) / (torch.sum(mask_inpaint.expand_as(replace_out)) + 1e-8))
            sim_loss = (torch.sum(torch.sum((torch.abs(edit_out.detach() - replace_out)), -1)[..., None] * mask_wo_edit) / (torch.sum(mask_wo_edit.expand_as(replace_out)) + 1e-8))
            
            # print(dissimilar_loss)

            correlation = torch.bmm(replace_out_att[:, mask_inpaint[0, 0, :, 0] > 0.5], edit_base_att.permute(0, -1, 1).detach())

            correlation_inpaint = correlation * mask_inpaint[..., 0]
            correlation_wo_edit = correlation * mask_wo_edit[..., 0]

            # p_correlation_inpaint = torch.max(correlation_inpaint, -1).values
            # p_correlation_wo_edit = torch.max(correlation_wo_edit, -1).values

            # loss_correlation = torch.sum(-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4)) / (torch.sum(mask_inpaint) * f + 1e-8)


            m_c_inpaint = torch.max(correlation_inpaint, -1)
            m_c_wo_edit = torch.max(correlation_wo_edit, -1)

            p_correlation_inpaint, d_inpaint = m_c_inpaint.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_inpaint.indices]
            p_correlation_wo_edit, d_wo_edit = m_c_wo_edit.values, distance_grid[:, mask_inpaint[0, 0, :, 0] > 0.5, m_c_wo_edit.indices]


            with torch.no_grad():
                d_weight = torch.exp(-d_wo_edit.detach())

            loss_correlation = torch.sum(d_weight.detach() * (-torch.log(p_correlation_wo_edit + 1e-4) + torch.log(p_correlation_inpaint + 1e-4))) / (torch.sum(mask_inpaint) * f + 1e-8)
            del m_c_inpaint, m_c_wo_edit
            
            # loss_correlation = torch.sum(torch.max(correlation * mask_inpaint[..., 0] / (torch.sum(correlation * mask_inpaint[..., 0], -1, keepdims=True) + 1e-8), -1).values - torch.max(correlation  * mask_wo_edit[..., 0] / (torch.sum(correlation * mask_wo_edit[..., 0], -1, keepdims=True) + 1e-8), -1).values) / (torch.sum(mask_inpaint) * f + 1e-8)

            # print(loss_correlation, "Self")        

            # print(replace_out.shape, " replace out")

            smoothness_loss, _, _ = get_smoothness_loss(replace_out)
            # print(smoothness_loss)

            # exit()

            dissociate_loss, att_loss, dissociate_loss_2, movement_loss = 0, 0, 0, 0.0


            lw = self.loss_weight_dict["self"]
            self.loss += (lw["sim"] * sim_loss + lw["removal"]* loss_correlation) + lw["smoothness"] * smoothness_loss

            loss_log_dict = {"sim": sim_loss, "removal": loss_correlation, "smoothness": smoothness_loss}
            self.loss_log_dict["self"] = update_loss_log_dict(self.loss_log_dict["self"], loss_log_dict) 
            self.loss_log_dict["num_layers"] += 1


            # self.loss += 0.5 * (0.0 * dissociate_loss + 0.0 * att_loss + 365.0 * sim_loss + 0.0 * movement_loss + 0.0 *  dissociate_loss_2 + 0.0 * dissimilar_loss + 35.0 * loss_correlation + 330 * smoothness_loss) / 3


        if self.cur_step < int(self.num_steps * self.obj_edit_step):
            out_return = replace_out * mask_inpaint + replace_out * mask_wo_edit
        else:
            out_return = replace_out_identity * mask_inpaint + replace_out * mask_wo_edit
        

        return out_return    


    def forward(self, q, k, v, is_cross: bool, place_in_unet: str, transform_coords = None, scale = None, mask = None):


        if self.use_cfg:
            h = q.shape[0] // (2 * self.batch_size)
        else:
            h = q.shape[0] // self.batch_size


        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            attn = compute_attention(q[:self.coords_base[-1] * h], k[:self.coords_base[-1] * h], scale, mask)
            out = torch.bmm(attn, v[:self.coords_base[-1] * h])
        else:
            attn = compute_attention(q, k, scale, mask)
            out = torch.bmm(attn, v)


        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):            
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                out_edit = self.replace_cross_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty = False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit, old_attention_map = attn, old_attention_out = out)
                
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

            else:
                out_edit = self.replace_self_attention(q, k, v, place_in_unet, transform_coords=transform_coords, add_empty=False, scale = scale, mask = mask, coords_base = self.coords_base, coords_edit = self.coords_edit, old_attention_map = attn, old_attention_out = out)
                out = torch.cat([out[:self.coords_base[-1] * h], out_edit[0]])

        return out
    

    def initialize_loss_log_dict(self):

        self.loss_log_dict = {"self": {"sim": 0.0, "removal": 0.0, "smoothness": 0.0},
                             "cross": {"sim": 0.0, "removal": 0.0, "smoothness": 0.0}, 
                             "num_layers": 0}
    
    def initialize_default_loss_weights(self):
        self.loss_weight_dict = self.default_loss_weights


    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer = None,
                local_blend = None, controller = None, image_mask = None, empty_scale = 0.2, use_all = True, obj_edit_step = 0.0):
        super(AttentionGeometryRemover, self).__init__()
        if equalizer is not None:
            eq = get_equalizer(prompts[1], equalizer["words"], equalizer["values"])
            self.equalizer = eq.to(DEVICE)
        self.prev_controller = controller
        self.last_cross_mask = None
        
        if image_mask is not None:
            image_mask = torch.from_numpy(image_mask[None])
            image_mask = image_mask.tile((len(prompts), 1, 1))
            self.image_mask = torch_dilate(image_mask[:, None], 5)[:, 0]
        self.thre = 0.00001
        self.empty_scale = empty_scale
        self.use_all = use_all
        self.loss = 0.0

        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(DEVICE)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.mask_inpaint = None
        self.obj_edit_step = obj_edit_step
        self.num_steps = num_steps

        self.mask_new_warped = None
        self.mask_wo_edit = None
        self.mask_1_empty = None
        self.amodal_mask = None

        self.coords_base = (2, 3)
        self.coords_edit = (3, 4)
        self.use_cfg = True

        self.loss_weight_dict = None
        # self.default_loss_weights = {"self": {"sim": 73, "removal": 2.25, "smoothness": 85.0},
        # "cross": {"sim": 41.4, "removal": 3.6, "smoothness": 45.0}}

        self.default_loss_weights = {"self": {"sim": 110.0, "removal": 3.6, "smoothness": 35.0},
        "cross": {"sim": 60.0, "removal": 3.6, "smoothness": 20.0}}

        # self.default_loss_weights = {"self": {"sim": 110, "movement": 13.5, "removal": 1.67, "smoothness": 35.0},
        #                         "cross": {"sim": 60, "movement": 6.34, "removal": 1.6, "smoothness": 20.0}}
        self.initialize_default_loss_weights()
        self.loss_log_dict = None
        self.initialize_loss_log_dict()
        
